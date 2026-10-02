from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neo4j import GraphDatabase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.dump.read_text(encoding="utf-8"))
    expected_nodes = len(payload["nodes"])
    expected_relationships = len(payload["relationships"])

    uri = os.environ.get("LOCAL_NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("LOCAL_NEO4J_USERNAME", "neo4j")
    password = os.environ.get("LOCAL_NEO4J_PASSWORD", "password")
    database = os.environ.get("LOCAL_NEO4J_DATABASE", "neo4j")

    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            """
            CALL {
              MATCH (n) RETURN count(n) AS nodes
            }
            CALL {
              MATCH ()-[r]->() RETURN count(r) AS relationships
            }
            RETURN nodes, relationships
            """,
            database_=database,
        )
        actual_nodes = int(records[0]["nodes"])
        actual_relationships = int(records[0]["relationships"])

        source_ids, _, _ = driver.execute_query(
            """
            MATCH (n)
            WHERE n.__source_element_id IS NOT NULL
            RETURN count(n) AS mapped
            """,
            database_=database,
        )

    assert actual_nodes == expected_nodes, (actual_nodes, expected_nodes)
    assert actual_relationships == expected_relationships, (
        actual_relationships,
        expected_relationships,
    )
    assert int(source_ids[0]["mapped"]) == expected_nodes

    print(
        f"Dump verification passed: {actual_nodes} nodes, "
        f"{actual_relationships} relationships"
    )


if __name__ == "__main__":
    main()
