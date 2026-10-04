from __future__ import annotations

from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index

from graphrag_drift.config import Neo4jSettings
from graphrag_drift.embedding import DEFAULT_EMBEDDING_DIMENSIONS, hash_embedding
from graphrag_drift.neo4j_graphrag_retrieval import (
    CHUNK_FULLTEXT_INDEX,
    CHUNK_VECTOR_INDEX,
    ENTITY_FULLTEXT_INDEX,
    ENTITY_VECTOR_INDEX,
)


def _embed_nodes(driver, *, database: str | None, label: str, text_expression: str) -> int:
    records, _, _ = driver.execute_query(
        f"""
        MATCH (n:{label})
        RETURN elementId(n) AS element_id,
               trim({text_expression}) AS text
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
            f"""
            UNWIND $rows AS row
            MATCH (n:{label}) WHERE elementId(n) = row.element_id
            SET n.embedding = row.embedding
            RETURN count(n) AS updated
            """,
            parameters_={"rows": rows},
            database_=database,
        )
    return len(rows)


def main() -> None:
    settings = Neo4jSettings.from_env()
    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        create_vector_index(
            driver,
            ENTITY_VECTOR_INDEX,
            label="Entity",
            embedding_property="embedding",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            similarity_fn="cosine",
            fail_if_exists=False,
            neo4j_database=settings.database,
        )
        create_fulltext_index(
            driver,
            ENTITY_FULLTEXT_INDEX,
            label="Entity",
            node_properties=["name", "description"],
            fail_if_exists=False,
            neo4j_database=settings.database,
        )
        create_vector_index(
            driver,
            CHUNK_VECTOR_INDEX,
            label="Chunk",
            embedding_property="embedding",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            similarity_fn="cosine",
            fail_if_exists=False,
            neo4j_database=settings.database,
        )
        create_fulltext_index(
            driver,
            CHUNK_FULLTEXT_INDEX,
            label="Chunk",
            node_properties=["text", "content"],
            fail_if_exists=False,
            neo4j_database=settings.database,
        )

        entity_count = _embed_nodes(
            driver,
            database=settings.database,
            label="Entity",
            text_expression="coalesce(n.name, '') + ' ' + coalesce(n.description, '')",
        )
        chunk_count = _embed_nodes(
            driver,
            database=settings.database,
            label="Chunk",
            text_expression="coalesce(n.text, n.content, '')",
        )
        driver.execute_query(
            "CALL db.awaitIndexes(300)",
            database_=settings.database,
        )
        print(
            "Vector setup complete and indexes ONLINE: "
            f"{entity_count} entities embedded, {chunk_count} chunks embedded"
        )


if __name__ == "__main__":
    main()
