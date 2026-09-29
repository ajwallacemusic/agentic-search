"""Plain-files backend: a directory (or in-memory documents) searched with BM25, regex,
metadata filters, and an optional local vector index."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from agentic_search.backends.base import (
    BackendError,
    DiscoverDetail,
    UnsupportedOperation,
    rrf_merge,
)
from agentic_search.backends.filters import matches
from agentic_search.core.types import (
    Aggregate,
    Capability,
    CollectionInfo,
    Content,
    Document,
    Fetch,
    FieldSpec,
    FieldType,
    Filter,
    FilterOnly,
    Hit,
    Hybrid,
    ImagePart,
    Lexical,
    Manifest,
    Modality,
    QueryOp,
    Regex,
    StructuredPart,
    TextPart,
    Vector,
    text_of,
)
from agentic_search.embedders.base import Embedder, cosine_scores, normalize_rows, supports

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".json", ".jsonl", ".csv", ".tsv",
                   ".html", ".xml", ".yaml", ".yml"}
IMAGE_EXTENSIONS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".webp": "image/webp"}
EMBED_BATCH = 64
SAMPLE_DISTINCT_MAX = 20
_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _is_iso_date(s: str) -> bool:
    if len(s) < 8:
        return False
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        return False


def infer_field_type(values: list[Any]) -> FieldType:
    vals = [v for v in values if v is not None]
    if not vals:
        return FieldType.KEYWORD
    if all(isinstance(v, bool) for v in vals):
        return FieldType.BOOL
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return FieldType.INT
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return FieldType.FLOAT
    if all(isinstance(v, str) for v in vals):
        return FieldType.DATE if all(_is_iso_date(v) for v in vals[:50]) else FieldType.KEYWORD
    return FieldType.JSON


class FilesBackend:
    backend_type = "files"

    def __init__(self, name: str, root: str | Path | None = None, *,
                 documents: Iterable[Document] | None = None, embedder: Embedder | None = None,
                 glob: str = "**/*", collection: str = "files", description: str | None = None,
                 max_file_bytes: int = 2_000_000):
        if (root is None) == (documents is None):
            raise ValueError("pass exactly one of root or documents")
        self.name = name
        self.root = Path(root) if root is not None else None
        self._given = list(documents) if documents is not None else None
        self.embedder = embedder
        self.glob = glob
        self.collection = collection
        self.description = description
        self.max_file_bytes = max_file_bytes
        self._docs: dict[str, Document] = {}
        self._ids: list[str] = []
        self._token_sets: list[set[str]] = []
        self._bm25: BM25Okapi | None = None
        self._vec_ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._lock = asyncio.Lock()
        self._loaded = False

    @classmethod
    def from_documents(cls, name: str, documents: Iterable[Document], **kwargs: Any) -> FilesBackend:
        return cls(name, documents=documents, **kwargs)

    # ---- loading ------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        async with self._lock:
            if self._loaded:
                return
            docs = self._given if self._given is not None else await asyncio.to_thread(self._scan)
            self._docs = {d.doc_id: d for d in docs}
            self._ids = list(self._docs)
            corpus = [tokenize(text_of(self._docs[i].content)) for i in self._ids]
            self._token_sets = [set(toks) for toks in corpus]
            if any(corpus):
                self._bm25 = BM25Okapi(corpus)
            if self.embedder is not None:
                await self._build_vectors()
            self._loaded = True

    def _scan(self) -> list[Document]:
        assert self.root is not None
        docs: list[Document] = []
        for path in sorted(self.root.glob(self.glob)):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            stat = path.stat()
            rel = path.relative_to(self.root).as_posix()
            meta = {"path": rel, "ext": ext, "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()}
            content: list[Content]
            if ext in TEXT_EXTENSIONS:
                if stat.st_size > self.max_file_bytes:
                    continue
                content = [TextPart(text=path.read_text(encoding="utf-8", errors="replace"))]
            elif ext in IMAGE_EXTENSIONS:
                content = [ImagePart(uri=path.resolve().as_uri(), mime=IMAGE_EXTENSIONS[ext])]
            else:
                continue
            docs.append(Document(doc_id=rel, content=content, metadata=meta))
        return docs

    def _embeddable_part(self, doc: Document) -> Content | None:
        assert self.embedder is not None
        text = text_of(doc.content)
        if text and supports(self.embedder, TextPart(text=text)):
            return TextPart(text=text)
        for part in doc.content:
            if isinstance(part, ImagePart) and supports(self.embedder, part):
                return part
        return None

    async def _build_vectors(self) -> None:
        assert self.embedder is not None
        items = [(i, p) for i in self._ids if (p := self._embeddable_part(self._docs[i])) is not None]
        vectors: list[list[float]] = []
        for start in range(0, len(items), EMBED_BATCH):
            batch = items[start:start + EMBED_BATCH]
            vectors.extend(await self.embedder.embed([p for _, p in batch], "document"))
        self._vec_ids = [i for i, _ in items]
        self._matrix = normalize_rows(np.asarray(vectors, dtype=np.float32)) if vectors else None

    # ---- protocol -----------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.LEXICAL, Capability.FILTER, Capability.REGEX, Capability.FETCH,
                Capability.AGGREGATE}
        if self.embedder is not None:
            caps |= {Capability.VECTOR, Capability.HYBRID}
            if Modality.IMAGE in self.embedder.modalities:
                caps.add(Capability.IMAGE_QUERY)
        return caps

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        await self._ensure_loaded()
        fields = [FieldSpec(name="text", type=FieldType.TEXT, searchable=True)]
        fields.extend(self._metadata_fields())
        if self.embedder is not None:
            fields.append(FieldSpec(name="embedding", type=FieldType.VECTOR,
                                    vector_dim=self.embedder.dim, vector_metric="cosine",
                                    embedder_id=self.embedder.id))
        return Manifest(
            source=self.name, backend_type=self.backend_type, capabilities=self.capabilities(),
            description=self.description,
            collections=[CollectionInfo(name=self.collection, fields=fields, count=len(self._docs))],
        )

    def _metadata_fields(self) -> list[FieldSpec]:
        values: dict[str, list[Any]] = {}
        for doc in self._docs.values():
            for k, v in doc.metadata.items():
                values.setdefault(k, []).append(v)
        out = []
        for key in sorted(values):
            ftype = infer_field_type(values[key])
            distinct = list(dict.fromkeys(
                v for v in values[key] if isinstance(v, (str, int, float, bool))))
            samples = (distinct if ftype in (FieldType.KEYWORD, FieldType.BOOL)
                       and len(distinct) <= SAMPLE_DISTINCT_MAX else None)
            out.append(FieldSpec(
                name=key, type=ftype, filterable=True,
                sortable=ftype in (FieldType.INT, FieldType.FLOAT, FieldType.DATE),
                sample_values=samples))
        return out

    async def execute(self, op: QueryOp) -> list[Hit]:
        await self._ensure_loaded()
        if op.collection not in (None, self.collection):
            raise BackendError(f"unknown collection {op.collection!r}; this source has {self.collection!r}")
        if isinstance(op, Lexical):
            ranked = self._lexical_ranking(op.text, set(self._allowed(op.filter)))
            return [self._hit(i, s) for i, s in ranked[: op.limit]]
        if isinstance(op, Vector):
            self._check_vector_field(op.field)
            ranked = self._vector_ranking(op.vector, set(self._allowed(op.filter)))
            return [self._hit(i, s) for i, s in ranked[: op.limit]]
        if isinstance(op, Hybrid):
            return self._hybrid(op)
        if isinstance(op, FilterOnly):
            return [self._hit(i) for i in self._allowed(op.filter)[: op.limit]]
        if isinstance(op, Regex):
            return self._regex(op)
        if isinstance(op, Fetch):
            return [self._hit(i) for i in op.doc_ids if i in self._docs]
        if isinstance(op, Aggregate):
            return self._aggregate(op)
        raise UnsupportedOperation(f"files backend does not support {op.type}")

    async def close(self) -> None:
        return None

    # ---- op implementations -------------------------------------------------

    def _allowed(self, f: Filter | None) -> list[str]:
        if f is None:
            return list(self._ids)
        return [i for i in self._ids if matches(f, {**self._docs[i].metadata, "doc_id": i})]

    def _hit(self, doc_id: str, score: float | None = None) -> Hit:
        d = self._docs[doc_id]
        return Hit(doc_id=doc_id, source=self.name, content=d.content, metadata=d.metadata,
                   raw_score=score)

    def _lexical_ranking(self, text: str, allowed: set[str]) -> list[tuple[str, float]]:
        query = tokenize(text)
        if self._bm25 is None or not query:
            return []
        qset = set(query)
        scores = self._bm25.get_scores(query)
        ranked = [(i, float(s)) for i, s, toks in zip(self._ids, scores, self._token_sets)
                  if i in allowed and qset & toks]
        ranked.sort(key=lambda kv: (-kv[1], kv[0]))
        return ranked

    def _check_vector_field(self, field: str) -> None:
        if self.embedder is None:
            raise BackendError("this source has no vector index")
        if field != "embedding":
            raise BackendError(f"unknown vector field {field!r}; this source has 'embedding'")

    def _vector_ranking(self, vector: list[float] | None, allowed: set[str]) -> list[tuple[str, float]]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        if self._matrix is None:
            return []
        scores = cosine_scores(vector, self._matrix)
        ranked = [(i, float(s)) for i, s in zip(self._vec_ids, scores) if i in allowed]
        ranked.sort(key=lambda kv: (-kv[1], kv[0]))
        return ranked

    def _hybrid(self, op: Hybrid) -> list[Hit]:
        self._check_vector_field(op.field)
        allowed = set(self._allowed(op.filter))
        depth = max(op.limit * 5, 50)
        lexical = [i for i, _ in self._lexical_ranking(op.text, allowed)][:depth]
        semantic = [i for i, _ in self._vector_ranking(op.vector, allowed)][:depth]
        fused = rrf_merge([(lexical, op.lexical_weight), (semantic, 1.0 - op.lexical_weight)])
        return [self._hit(i, s) for i, s in fused[: op.limit]]

    def _regex(self, op: Regex) -> list[Hit]:
        try:
            pattern = re.compile(op.pattern)
        except re.error as exc:
            raise BackendError(f"invalid regex: {exc}") from exc
        fields = op.fields or ["text", "path"]
        out: list[Hit] = []
        for i in self._allowed(op.filter):
            doc = self._docs[i]
            for f in fields:
                value = text_of(doc.content) if f == "text" else doc.metadata.get(f)
                if value is not None and pattern.search(str(value)):
                    out.append(self._hit(i))
                    break
            if len(out) >= op.limit:
                break
        return out

    def _aggregate(self, op: Aggregate) -> list[Hit]:
        unsupported = [m for m in op.metrics if m != "count"]
        if unsupported:
            raise UnsupportedOperation(f"files backend only supports the 'count' metric, not {unsupported}")
        groups: dict[str, tuple[dict[str, Any], int]] = {}
        for i in self._allowed(op.filter):
            row = {g: self._docs[i].metadata.get(g) for g in op.group_by}
            key = json.dumps(row, sort_keys=True, default=str)
            groups[key] = (row, groups.get(key, (row, 0))[1] + 1)
        ranked = sorted(groups.items(), key=lambda kv: (-kv[1][1], kv[0]))[: op.limit]
        return [Hit(doc_id=f"agg:{key}", source=self.name,
                    content=[StructuredPart(data={**row, "count": n})], raw_score=float(n))
                for key, (row, n) in ranked]
