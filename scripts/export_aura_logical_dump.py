from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from graphrag_drift.config import Neo4jSettings


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def export_graph(path: Path) -> dict[str, int]:
    settings = Neo4jSettings.from_env()
    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        node_records, _, _ = driver.execute_query(
            """
            MATCH (n)
            RETURN elementId(n) AS source_id,
                   labels(n) AS labels,
                   properties(n) AS properties
            ORDER BY source_id
            """,
            database_=settings.database,
        )
        rel_records, _, _ = driver.execute_query(
            """
            MATCH (a)-[r]->(b)
            RETURN elementId(r) AS source_id,
                   elementId(a) AS start_id,
                   elementId(b) AS end_id,
                   type(r) AS type,
                   properties(r) AS properties
            ORDER BY source_id
            """,
            database_=settings.database,
        )

    payload = {
        "format": "graphrag-drift-logical-dump-v1",
        "nodes": [
            {
                "source_id": record["source_id"],
                "labels": list(record["labels"]),
                "properties": json_safe(record["properties"]),
            }
            for record in node_records
        ],
        "relationships": [
            {
                "source_id": record["source_id"],
                "start_id": record["start_id"],
                "end_id": record["end_id"],
                "type": record["type"],
                "properties": json_safe(record["properties"]),
            }
            for record in rel_records
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "nodes": len(payload["nodes"]),
        "relationships": len(payload["relationships"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    counts = export_graph(args.output)
    print(
        f"Aura logical dump complete: {counts['nodes']} nodes, "
        f"{counts['relationships']} relationships"
    )


if __name__ == "__main__":
    main()
