from __future__ import annotations

from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index

from graphrag_drift.config import Neo4jSettings
from graphrag_drift.embedding import DEFAULT_EMBEDDING_DIMENSIONS, hash_embedding
from graphrag_drift.neo4j_graphrag_retrieval import ENTITY_FULLTEXT_INDEX, ENTITY_VECTOR_INDEX


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

        records, _, _ = driver.execute_query(
            """
            MATCH (e:Entity)
            RETURN elementId(e) AS element_id,
                   trim(coalesce(e.name, '') + ' ' + coalesce(e.description, '')) AS text
            """,
            database_=settings.database,
        )
        rows = [
            {"element_id": record["element_id"], "embedding": hash_embedding(record["text"])}
            for record in records
        ]
        driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (e:Entity) WHERE elementId(e) = row.element_id
            SET e.embedding = row.embedding
            RETURN count(e) AS updated
            """,
            parameters_={"rows": rows},
            database_=settings.database,
        )
        print(f"Vector setup complete: {len(rows)} entities embedded")


if __name__ == "__main__":
    main()
