"""Seed the Neo4j test service (separate write session) and build the backend under test."""

from __future__ import annotations

from . import corpus

NEO4J_URI = "bolt://localhost:57687"
NEO4J_AUTH = ("neo4j", "agenticpass")


async def seed_neo4j() -> None:
    import neo4j

    vectors = await corpus.embeddings()
    driver = neo4j.AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, notifications_min_severity="OFF")
    try:
        async with driver.session() as s:
            await (await s.run("MATCH (n) DETACH DELETE n")).consume()
            for row in await (await s.run("SHOW INDEXES YIELD name, type WHERE type IN ['FULLTEXT', 'VECTOR'] "
                                          "RETURN name")).data():
                await (await s.run(f"DROP INDEX `{row['name']}`")).consume()
            for (i, t, b, ty, y), v in zip(corpus.ROWS, vectors):
                await (await s.run("CREATE (:docs {id: $id, title: $t, body: $b, type: $ty, year: $y, "
                                   "embedding: $v})", id=i, t=t, b=b, ty=ty, y=y, v=v)).consume()
            await (await s.run("CREATE (:conditions {id: 'headache'})")).consume()
            await (await s.run("MATCH (d:docs), (c:conditions {id: 'headache'}) WHERE d.id IN ['d1', 'd4'] "
                               "CREATE (d)-[:TREATS]->(c)")).consume()
            await (await s.run("CREATE FULLTEXT INDEX docs_text FOR (n:docs) ON EACH [n.title, n.body]")).consume()
            await (await s.run("CREATE VECTOR INDEX docs_embedding FOR (n:docs) ON n.embedding OPTIONS "
                               "{indexConfig: {`vector.dimensions`: 64, `vector.similarity_function`: 'cosine'}}")
                   ).consume()
            await (await s.run("CALL db.awaitIndexes(300)")).consume()
    finally:
        await driver.close()


async def make_neo4j():
    from agentic_search.backends.neo4j import Neo4jBackend

    await seed_neo4j()
    return Neo4jBackend("neo", NEO4J_URI, user=NEO4J_AUTH[0], password=NEO4J_AUTH[1],
                        labels=["docs", "conditions"], embedders={"docs.embedding": "hash64"},
                        native_query=True)
