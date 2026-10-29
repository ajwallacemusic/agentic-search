import json

from demo.ignite.public.datasets import para_id, sample_musique


def _para(idx, title, text, supporting=False):
    return {"idx": idx, "title": title, "paragraph_text": text, "is_supporting": supporting}


def _record(qid, question, paras):
    return {"id": qid, "question": question, "answer": "x", "answer_aliases": [],
            "answerable": True, "question_decomposition": [], "paragraphs": paras}


def _musique(tmp_path, n_per_hop=3):
    rows = []
    for hop in ("2hop", "3hop1", "4hop1"):
        for i in range(n_per_hop):
            qid = f"{hop}__{i}"
            paras = [_para(j, f"{qid} title {j}", f"{qid} text {j}", supporting=j < 2)
                     for j in range(5)]
            rows.append(_record(qid, f"question {qid}?", paras))
    path = tmp_path / "musique_ans_v1.0_dev.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, rows


def test_samples_per_hop_group(tmp_path):
    path, _ = _musique(tmp_path)
    s = sample_musique(path, per_hop=2, corpus_size=100)
    assert {g: len(q) for g, q in s.groups.items()} == {"2hop": 2, "3hop": 2, "4hop": 2}
    assert set(s.queries) == {q for qs in s.groups.values() for q in qs}


def test_gold_is_the_supporting_paragraphs(tmp_path):
    path, rows = _musique(tmp_path)
    s = sample_musique(path, per_hop=2, corpus_size=100)
    by_id = {r["id"]: r for r in rows}
    ids = {d.doc_id for d in s.documents}
    for qid, gold in s.qrels.items():
        want = {para_id(p["title"], p["paragraph_text"]) for p in by_id[qid]["paragraphs"]
                if p["is_supporting"]}
        assert set(gold) == want and set(gold.values()) == {1} and set(gold) <= ids


def test_shared_paragraph_keeps_one_id_and_gold_follows(tmp_path):
    path, rows = _musique(tmp_path, n_per_hop=1)
    shared = _para(0, "Shared", "Same text in two questions.", supporting=True)
    rows[0]["paragraphs"][0] = shared
    rows[1]["paragraphs"][0] = {**shared, "is_supporting": True}
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = sample_musique(path, per_hop=1, corpus_size=100)
    sid = para_id("Shared", "Same text in two questions.")
    assert [d.doc_id for d in s.documents].count(sid) == 1
    assert sid in s.qrels[rows[0]["id"]] and sid in s.qrels[rows[1]["id"]]


def test_cap_never_drops_sampled_paragraphs(tmp_path):
    path, _ = _musique(tmp_path)
    s = sample_musique(path, per_hop=2, corpus_size=5)  # below the 6 x 5 sampled paragraphs
    assert len(s.documents) == 30
    ids = {d.doc_id for d in s.documents}
    assert all(set(g) <= ids for g in s.qrels.values())


def test_distractors_fill_to_the_cap(tmp_path):
    path, _ = _musique(tmp_path)
    s = sample_musique(path, per_hop=1, corpus_size=40)
    assert len(s.documents) == 40  # 15 sampled + 25 from unsampled questions


def test_musique_sample_is_deterministic(tmp_path):
    path, _ = _musique(tmp_path)
    a = sample_musique(path, per_hop=2, corpus_size=40, seed=3)
    b = sample_musique(path, per_hop=2, corpus_size=40, seed=3)
    assert a.queries == b.queries and [d.doc_id for d in a.documents] == [d.doc_id for d in b.documents]


def test_musique_sample_converts_to_beir(tmp_path):
    path, _ = _musique(tmp_path)
    ds = sample_musique(path, per_hop=1, corpus_size=20).to_beir()
    assert len(ds.corpus) == 20 and len(ds.queries) == 3 and len(ds.qrels) == 3


# ---- WANDS ------------------------------------------------------------------------------

from demo.ignite.public.datasets import sample_wands  # noqa: E402

PRODUCT_HEADER = ("product_id\tproduct_name\tproduct_class\tcategory hierarchy\tproduct_description"
                  "\tproduct_features\trating_count\taverage_rating\treview_count\n")


def _wands(tmp_path):
    d = tmp_path / "wands"
    d.mkdir()
    products = [
        "0\tsolid wood bed\tBeds\tFurniture / Beds\ta sturdy bed\tcolor:brown\t10\t4.5\t8",
        "1\tvelvet sofa\tSofas\tFurniture / Sofas\t\t\t\t\t",                       # empty fields
        '2\tglass lamp\tLamps\tLighting / Lamps\t"a lamp\twith a tab"\tsize:small\t3\t4.0\t2',
        "3\toak table\tTables\tFurniture / Tables\tsolid oak\t\t5\t3.9\t5",
    ]
    (d / "product.csv").write_text(PRODUCT_HEADER + "\n".join(products) + "\n")
    (d / "query.csv").write_text("query_id\tquery\tquery_class\n0\twood bed\tBeds\n"
                                 "1\tsofa\tSofas\n2\tlamp\tLamps\n3\tnothing exact\tMisc\n")
    (d / "label.csv").write_text("id\tquery_id\tproduct_id\tlabel\n"
                                 "0\t0\t0\tExact\n1\t0\t3\tPartial\n2\t0\t1\tIrrelevant\n"
                                 "3\t1\t1\tExact\n4\t2\t2\tExact\n5\t3\t0\tPartial\n")
    return d


def test_only_queries_with_an_exact_label(tmp_path):
    s = sample_wands(_wands(tmp_path), n=10)
    assert set(s.queries) == {"0", "1", "2"}
    assert s.groups == {"all": sorted(s.queries)}


def test_labels_are_graded(tmp_path):
    s = sample_wands(_wands(tmp_path), n=10)
    assert s.qrels["0"] == {"wands-0": 2, "wands-3": 1}


def test_products_with_empty_fields_load(tmp_path):
    s = sample_wands(_wands(tmp_path), n=10)
    docs = {d.doc_id: d for d in s.documents}
    assert set(docs) == {"wands-0", "wands-1", "wands-2", "wands-3"}
    assert docs["wands-1"].metadata["product_class"] == "Sofas"
    assert "rating_count" not in docs["wands-1"].metadata
    assert docs["wands-0"].metadata["average_rating"] == 4.5


def test_wands_sample_is_deterministic(tmp_path):
    folder = _wands(tmp_path)
    assert sample_wands(folder, n=2, seed=5).queries == sample_wands(folder, n=2, seed=5).queries
    assert len(sample_wands(folder, n=2).queries) == 2
