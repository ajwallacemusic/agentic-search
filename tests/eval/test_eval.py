import json

import pytest

from agentic_search import Harness
from agentic_search.backends.files import FilesBackend
from agentic_search.eval.datasets import load_beir
from agentic_search.eval.metrics import mrr, ndcg_at_k, recall_at_k
from agentic_search.eval.runner import run_eval
from agentic_search.testing import EchoDriver


def test_metrics():
    assert ndcg_at_k(["a", "b"], {"a": 1}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], {"a": 1}, 10) == pytest.approx(0.6309, abs=1e-4)
    assert ndcg_at_k(["x"], {}, 10) == 0.0
    assert recall_at_k(["a", "x"], {"a": 1, "b": 2, "c": 0}, 1) == 0.5
    assert mrr(["x", "a"], {"a": 1}) == 0.5 and mrr(["x"], {"a": 1}) == 0.0


@pytest.fixture
def beir_dir(tmp_path):
    corpus = [{"_id": "d1", "title": "", "text": "aspirin headache"},
              {"_id": "d2", "title": "Castles", "text": "castle history"},
              {"_id": "d3", "title": "", "text": "ibuprofen headache pain"}]
    queries = [{"_id": "q1", "text": "headache"}, {"_id": "q2", "text": "castle"},
               {"_id": "q3", "text": "no qrels"}]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in corpus))
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in queries))
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t2\nq2\td2\t1\n")
    return tmp_path


def test_load_beir(beir_dir):
    ds = load_beir(beir_dir)
    assert set(ds.queries) == {"q1", "q2"} and ds.qrels["q1"] == {"d1": 1, "d3": 2}
    docs = {d.doc_id: d for d in ds.documents()}
    assert docs["d2"].content[0].text == "Castles\ncastle history"
    assert docs["d2"].metadata == {"title": "Castles"} and docs["d1"].metadata == {}


async def test_run_eval_bm25_baseline(beir_dir):
    ds = load_beir(beir_dir)
    backend = FilesBackend.from_documents("corpus", ds.documents())
    h = Harness([backend], EchoDriver("corpus"), mode="retrieval")
    report = await run_eval(h, ds, name="bm25")
    runs = {r.query_id: r for r in report.runs}
    # q1 ranks d1 (short doc) above d3: DCG = 1 + 2/log2(3); IDCG = 2 + 1/log2(3)
    assert runs["q1"].ndcg == pytest.approx(0.8597, abs=1e-3)
    assert runs["q2"].ndcg == pytest.approx(1.0) and runs["q2"].recall == 1.0
    assert report.mean("ndcg") == pytest.approx((0.8597 + 1.0) / 2, abs=1e-3)
    assert "bm25" in report.table() and "nDCG@10" in report.table()
    assert all(r.error is None for r in report.runs)


def test_eval_script_retrieval_baseline_is_unjudged_unless_reranked():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_beir.py"
    spec = importlib.util.spec_from_file_location("eval_beir", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    judge = object()
    assert mod.analyzer_for("retrieval", judge, rerank_retrieval=False) is None
    assert mod.analyzer_for("retrieval", judge, rerank_retrieval=True) is judge
    assert mod.analyzer_for("harness", judge, rerank_retrieval=False) is judge
    assert mod.analyzer_for("model", judge, rerank_retrieval=False) is judge
