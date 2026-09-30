"""Seed the Milvus test collection `docs` from the shared corpus (test-only writes)."""

from __future__ import annotations

from . import corpus

MILVUS_URI = "http://localhost:59530"


async def seed_milvus() -> None:
    from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType, MilvusClient

    vectors = await corpus.embeddings()
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field("body", DataType.VARCHAR, max_length=4096, enable_analyzer=True)
    schema.add_field("type", DataType.VARCHAR, max_length=32)
    schema.add_field("year", DataType.INT64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=64)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="body_bm25", function_type=FunctionType.BM25,
                                 input_field_names=["body"], output_field_names=["sparse"]))
    index = MilvusClient.prepare_index_params()
    index.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                    params={"M": 16, "efConstruction": 64})
    index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client = AsyncMilvusClient(uri=MILVUS_URI)
    try:
        if "docs" in await client.list_collections():
            await client.drop_collection("docs")
        await client.create_collection("docs", schema=schema, index_params=index,
                                       consistency_level="Strong")
        rows = [{"id": i, "title": t, "body": b, "type": ty, "year": y, "embedding": v}
                for (i, t, b, ty, y), v in zip(corpus.ROWS, vectors)]
        await client.insert("docs", rows)
        await client.flush("docs")
        # The backend never loads collections by default (load_collections=False), so the seed does.
        await client.load_collection("docs")
    finally:
        await client.close()


_seeded = False


async def make_milvus():
    """Seed once per test process (contract tests never write), then return a fresh backend."""
    global _seeded
    from agentic_search.backends.milvus import MilvusBackend

    if not _seeded:
        await seed_milvus()
        _seeded = True
    return MilvusBackend("mv", MILVUS_URI, collections=["docs"], embedders={"docs.embedding": "hash64"})
