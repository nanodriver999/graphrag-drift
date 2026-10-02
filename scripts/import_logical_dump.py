from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from graphrag_drift.dump_utils import quote_identifier


def load_target() -> tuple[str, str, str, str]:
    return (
        os.environ.get("LOCAL_NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("LOCAL_NEO4J_USERNAME", "neo4j"),
        os.environ.get("LOCAL_NEO4J_PASSWORD", "password"),
        os.environ.get("LOCAL_NEO4J_DATABASE", "neo4j"),
    )


def import_graph(path: Path, *, clear: bool = True) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "graphrag-drift-logical-dump-v1":
        raise ValueError("Unsupported dump format")

    uri, username, password, database = load_target()
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()

        if clear:
            driver.execute_query("MATCH (n) DETACH DELETE n", database_=database)

        grouped_nodes: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for node in payload["nodes"]:
            labels = tuple(sorted(node["labels"]))
            grouped_nodes[labels].append(node)

        for labels, rows in grouped_nodes.items():
            label_clause = "".join(f":{quote_identifier(label)}" for label in labels)
            driver.execute_query(
                f"""
                UNWIND $rows AS row
                CREATE (n{label_clause})
                SET n = row.properties,
                    n.__source_element_id = row.source_id
                """,
                parameters_={"rows": rows},
                database_=database,
            )

        grouped_rels: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rel in payload["relationships"]:
            grouped_rels[rel["type"]].append(rel)

        for rel_type, rows in grouped_rels.items():
            rel_identifier = quote_identifier(rel_type)
            driver.execute_query(
                f"""
                UNWIND $rows AS row
                MATCH (a {{__source_element_id: row.start_id}})
                MATCH (b {{__source_element_id: row.end_id}})
                CREATE (a)-[r:{rel_identifier}]->(b)
                SET r = row.properties,
                    r.__source_element_id = row.source_id
                """,
                parameters_={"rows": rows},
                database_=database,
            )

        counts, _, _ = driver.execute_query(
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

    return {
        "nodes": int(counts[0]["nodes"]),
        "relationships": int(counts[0]["relationships"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    counts = import_graph(args.input, clear=not args.no_clear)
    print(
        f"Docker Neo4j import complete: {counts['nodes']} nodes, "
        f"{counts['relationships']} relationships"
    )


if __name__ == "__main__":
    main()
