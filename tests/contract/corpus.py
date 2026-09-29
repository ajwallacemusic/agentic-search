"""The shared fixture corpus every backend is seeded with, plus per-service seeding helpers.
Seeding is test-only and uses separate write connections; backends themselves are read-only."""

from __future__ import annotations

import json

from agentic_search.core.types import Document, TextPart
from agentic_search.embedders.local import HashEmbedder

EMBEDDER = HashEmbedder(dim=64, id="hash64")
ROWS = [
    ("d1", "Aspirin", "Aspirin reduces fever and relieves headache pain", "drug", 2020),
    ("d2", "Ibuprofen", "Ibuprofen is an anti-inflammatory used for pain", "drug", 2021),
    ("d3", "Printing press", "The history of the printing press in Europe", "history", 1999),
    ("d4", "Acetaminophen", "Acetaminophen treats headache and fever", "drug", 2019),
    ("d5", "Castles", "Medieval castles and their architecture", "history", 2005),
]
PG_DSN = "postgresql://postgres:agentic@localhost:55432/agentic"
MYSQL_DSN = "mysql://root:agentic@127.0.0.1:53306/agentic"
OPENSEARCH_URL = "http://localhost:59200"


async def embeddings() -> list[list[float]]:
    return await EMBEDDER.embed([TextPart(text=f"{t}\n{b}") for _, t, b, _, _ in ROWS], "document")


def documents() -> list[Document]:
    return [Document(doc_id=i, content=[TextPart(text=f"{t}\n{b}")],
                     metadata={"title": t, "type": ty, "year": y}) for i, t, b, ty, y in ROWS]


async def seed_postgres() -> None:
    import psycopg

    vectors = await embeddings()
    async with await psycopg.AsyncConnection.connect(PG_DSN, autocommit=True) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("DROP TABLE IF EXISTS docs")
        await conn.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, title TEXT, body TEXT, "
                           "type VARCHAR(20), year INT, embedding vector(64))")
        for (i, t, b, ty, y), v in zip(ROWS, vectors):
            await conn.execute("INSERT INTO docs VALUES (%s, %s, %s, %s, %s, %s::vector)",
                               [i, t, b, ty, y, "[" + ",".join(map(str, v)) + "]"])
        await conn.execute("ANALYZE docs")


async def seed_mysql() -> None:
    import aiomysql

    conn = await aiomysql.connect(host="127.0.0.1", port=53306, user="root", password="agentic",
                                  db="agentic", autocommit=True)
    try:
        async with conn.cursor() as cur:
            await cur.execute("SET sql_notes = 0")
            await cur.execute("DROP TABLE IF EXISTS docs")
            await cur.execute("CREATE TABLE docs (id VARCHAR(20) PRIMARY KEY, title VARCHAR(200), "
                              "body TEXT, type VARCHAR(20), year INT, "
                              "FULLTEXT KEY ft_docs (title, body)) ENGINE=InnoDB")
            await cur.executemany("INSERT INTO docs VALUES (%s, %s, %s, %s, %s)", ROWS)
            await cur.execute("ANALYZE TABLE docs")
    finally:
        conn.close()


async def seed_opensearch() -> None:
    from opensearchpy import AsyncOpenSearch

    vectors = await embeddings()
    client = AsyncOpenSearch(hosts=[OPENSEARCH_URL])
    try:
        if await client.indices.exists(index="docs"):
            await client.indices.delete(index="docs")
        await client.indices.create(index="docs", body={
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {"properties": {
                "title": {"type": "text"}, "body": {"type": "text"},
                "type": {"type": "keyword"}, "year": {"type": "integer"},
                "embedding": {"type": "knn_vector", "dimension": 64, "method": {
                    "name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}}}}})
        lines = []
        for (i, t, b, ty, y), v in zip(ROWS, vectors):
            lines.append(json.dumps({"index": {"_index": "docs", "_id": i}}))
            lines.append(json.dumps({"title": t, "body": b, "type": ty, "year": y, "embedding": v}))
        await client.bulk(body="\n".join(lines) + "\n", refresh=True)
    finally:
        await client.close()
