from __future__ import annotations

import argparse
import json

from neo4j import GraphDatabase

from graphrag_drift.config import Neo4jSettings
from graphrag_drift.temporal_coverage import LawVersion, evaluate_law_coverage, normalize_date


QUERY = """
MATCH (chunk:Chunk)
WHERE chunk.lawId IN $law_ids
RETURN DISTINCT
       toString(chunk.lawId) AS law_id,
       coalesce(chunk.lawTitle, '') AS title,
       coalesce(chunk.officialEffectiveDate, '') AS effective_date,
       coalesce(chunk.lawVersionKey, '') AS version_key
ORDER BY law_id, effective_date
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report whether required law IDs have a version effective on an as-of date."
    )
    parser.add_argument("--as-of", default=None)
    parser.add_argument(
        "--law-id",
        action="append",
        dest="law_ids",
        required=True,
        help="Required law ID. Repeat for multiple laws.",
    )
    parser.add_argument(
        "--require-current",
        action="store_true",
        help="Exit nonzero when any requested law has no current-effective version.",
    )
    args = parser.parse_args()

    as_of = normalize_date(args.as_of)
    settings = Neo4jSettings.from_env()
    driver = GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    )
    try:
        records, _, _ = driver.execute_query(
            QUERY,
            parameters_={"law_ids": args.law_ids},
            database_=settings.database,
        )
    finally:
        driver.close()

    versions = [
        LawVersion(
            law_id=str(record["law_id"]),
            title=str(record["title"] or ""),
            effective_date=str(record["effective_date"] or ""),
            version_key=str(record["version_key"] or ""),
        )
        for record in records
    ]

    results = []
    missing = []
    for law_id in args.law_ids:
        coverage = evaluate_law_coverage(law_id, versions, as_of_date=as_of)
        latest = coverage.latest_current
        row = {
            "law_id": law_id,
            "as_of_date": as_of,
            "has_current_version": coverage.has_current_version,
            "latest_current": None
            if latest is None
            else {
                "title": latest.title,
                "effective_date": latest.effective_date,
                "version_key": latest.version_key,
            },
            "future_versions": [
                {
                    "title": item.title,
                    "effective_date": item.effective_date,
                    "version_key": item.version_key,
                }
                for item in coverage.future_versions
            ],
        }
        results.append(row)
        if not coverage.has_current_version:
            missing.append(law_id)

    print(
        json.dumps(
            {
                "format": "graphrag-drift-temporal-coverage-v1",
                "as_of_date": as_of,
                "results": results,
                "missing_current_law_ids": missing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.require_current and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
