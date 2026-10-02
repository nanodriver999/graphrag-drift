from __future__ import annotations

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from graphrag_drift.config import Neo4jSettings


GRAPH_NAME = "graphrag_entity_projection"


def main() -> None:
    settings = Neo4jSettings.from_env()
    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        try:
            records, _, _ = driver.execute_query(
                "SHOW PROCEDURES YIELD name WHERE name = 'gds.leiden.write' RETURN name",
                database_=settings.database,
            )
        except ClientError as exc:
            print(f"Leiden unavailable: unable to inspect GDS procedures ({exc.code})")
            return

        if not records:
            print("Leiden unavailable: gds.leiden.write is not installed on this AuraDB instance")
            return

        try:
            driver.execute_query(
                "CALL gds.graph.drop($name, false)",
                parameters_={"name": GRAPH_NAME},
                database_=settings.database,
            )
        except ClientError:
            pass

        driver.execute_query(
            """
            CALL gds.graph.project(
              $name,
              'Entity',
              {USES: {orientation: 'UNDIRECTED'}}
            )
            """,
            parameters_={"name": GRAPH_NAME},
            database_=settings.database,
        )
        result, _, _ = driver.execute_query(
            """
            CALL gds.leiden.write($name, {writeProperty: 'leidenCommunity'})
            YIELD communityCount, nodePropertiesWritten
            RETURN communityCount, nodePropertiesWritten
            """,
            parameters_={"name": GRAPH_NAME},
            database_=settings.database,
        )
        driver.execute_query(
            """
            MATCH (e:Entity)
            WHERE e.leidenCommunity IS NOT NULL
            MERGE (c:Community {id: 'leiden-' + toString(e.leidenCommunity)})
            SET c.name = 'Leiden Community ' + toString(e.leidenCommunity)
            MERGE (e)-[:IN_COMMUNITY]->(c)
            """,
            database_=settings.database,
        )
        driver.execute_query(
            "CALL gds.graph.drop($name, false)",
            parameters_={"name": GRAPH_NAME},
            database_=settings.database,
        )
        print(
            "Leiden complete: "
            f"{result[0]['communityCount']} communities, "
            f"{result[0]['nodePropertiesWritten']} properties written"
        )


if __name__ == "__main__":
    main()
