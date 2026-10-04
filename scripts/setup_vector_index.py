from __future__ import annotations

from neo4j import GraphDatabase

from graphrag_drift.config import Neo4jSettings
from graphrag_drift.embedding import DEFAULT_EMBEDDING_DIMENSIONS, hash_embedding
from graphrag_drift.neo4j_graphrag_retrieval import (
    CHUNK_FULLTEXT_INDEX,
    ENTITY_FULLTEXT_INDEX,
    ENTITY_VECTOR_INDEX,
)


def _embed_entities(driver, *, database: str | None) -> int:
    records, _, _ = driver.execute_query(
        """
        MATCH (n:Entity)
        RETURN elementId(n) AS element_id,
               trim(coalesce(n.name, '') + ' ' + coalesce(n.description, '')) AS text
        """,
        database_=database,
    )
    rows = [
        {"element_id": record["element_id"], "embedding": hash_embedding(record["text"])}
        for record in records
        if str(record["text"] or "").strip()
    ]
    if rows:
        driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (n:Entity) WHERE elementId(n) = row.element_id
            SET n.embedding = row.embedding
            RETURN count(n) AS updated
            """,
            parameters_={"rows": rows},
            database_=database,
        )
    return len(rows)


def _create_indexes(driver, *, database: str | None) -> None:
    statements = [
        f"""
        CREATE VECTOR INDEX {ENTITY_VECTOR_INDEX} IF NOT EXISTS
        FOR (n:Entity) ON (n.embedding)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {DEFAULT_EMBEDDING_DIMENSIONS},
          `vector.similarity_function`: 'cosine'
        }}}}
        """,
        f"""
        CREATE FULLTEXT INDEX {ENTITY_FULLTEXT_INDEX} IF NOT EXISTS
        FOR (n:Entity) ON EACH [n.name, n.description]
        """,
        f"""
        CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS
        FOR (n:Chunk) ON EACH [n.text, n.content]
        """,
    ]
    for statement in statements:
        driver.execute_query(statement, database_=database)


def main() -> None:
    settings = Neo4jSettings.from_env()
    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        entity_count = _embed_entities(driver, database=settings.database)
        chunk_records, _, _ = driver.execute_query(
            "MATCH (n:Chunk) RETURN count(n) AS count",
            database_=settings.database,
        )
        chunk_count = int(chunk_records[0]["count"]) if chunk_records else 0

        _create_indexes(driver, database=settings.database)
        driver.execute_query(
            "CALL db.awaitIndexes(300)",
            database_=settings.database,
        )

        expected = {
            ENTITY_VECTOR_INDEX,
            ENTITY_FULLTEXT_INDEX,
            CHUNK_FULLTEXT_INDEX,
        }
        records, _, _ = driver.execute_query(
            """
            SHOW INDEXES
            YIELD name, type, state, labelsOrTypes, properties
            WHERE name IN $expected
            RETURN name, type, state, labelsOrTypes, properties
            ORDER BY name
            """,
            parameters_={"expected": sorted(expected)},
            database_=settings.database,
        )
        found = {str(record["name"]): record.data() for record in records}
        missing = expected - set(found)
        if missing:
            raise RuntimeError(f"Missing retrieval indexes after setup: {sorted(missing)}")

        not_online = [
            name for name, row in found.items() if str(row["state"]).upper() != "ONLINE"
        ]
        if not_online:
            raise RuntimeError(f"Retrieval indexes are not ONLINE: {not_online}")

        print(
            "Retrieval setup complete and indexes ONLINE: "
            f"{entity_count} entities embedded, {chunk_count} chunks available"
        )
        for name in sorted(found):
            print("index:", found[name])


if __name__ == "__main__":
    main()
