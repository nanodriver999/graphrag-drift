from __future__ import annotations

import argparse
import json

from neo4j import GraphDatabase

from graphrag_drift.config import Neo4jSettings
from graphrag_drift.temporal_coverage import (
    ArticleVersion,
    LawVersion,
    evaluate_article_coverage,
    evaluate_law_coverage,
    normalize_date,
)


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


ARTICLE_QUERY = """
MATCH (chunk:Chunk)
WHERE chunk.lawId IN $law_ids
  AND any(article IN $articles WHERE chunk.articleKey ENDS WITH ':art:' + article)
RETURN DISTINCT
       toString(chunk.lawId) AS law_id,
       coalesce(chunk.articleKey, '') AS article_key,
       coalesce(chunk.officialEffectiveDate, '') AS effective_date
ORDER BY law_id, article_key, effective_date
"""


def _parse_article_ids(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for item in values:
        law_id, sep, article = item.partition(":")
        if not sep or not law_id or not article:
            raise ValueError(
                f"Invalid --article {item!r}; expected law-id:article-fragment"
            )
        parsed.append((law_id, article))
    return parsed


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
        "--article",
        action="append",
        default=[],
        dest="articles",
        help="Required article as law-id:article-fragment, e.g. 011435:53-3.",
    )
    parser.add_argument(
        "--require-current",
        action="store_true",
        help="Exit nonzero when any requested law has no current-effective version.",
    )
    args = parser.parse_args()

    as_of = normalize_date(args.as_of)
    try:
        requested_articles = _parse_article_ids(args.articles)
    except ValueError as exc:
        parser.error(str(exc))
    if any(law_id not in args.law_ids for law_id, _ in requested_articles):
        parser.error("--article law ID must also be listed with --law-id")
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
        article_records = []
        if requested_articles:
            article_records, _, _ = driver.execute_query(
                ARTICLE_QUERY,
                parameters_={
                    "law_ids": list({law for law, _ in requested_articles}),
                    "articles": list({article for _, article in requested_articles}),
                },
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

    article_versions = [
        ArticleVersion(
            law_id=str(record["law_id"]),
            article=str(record["article_key"]).rsplit(":art:", 1)[-1],
            effective_date=str(record["effective_date"] or ""),
            article_key=str(record["article_key"] or ""),
        )
        for record in article_records
    ]
    article_results = []
    missing_articles = []
    for law_id, article in requested_articles:
        coverage = evaluate_article_coverage(
            law_id, article, article_versions, as_of_date=as_of,
        )
        current = coverage.latest_current
        article_results.append({
            "law_id": law_id,
            "article": article,
            "as_of_date": as_of,
            "has_current_article": coverage.has_current_article,
            "latest_current": None if current is None else {
                "article_key": current.article_key,
                "effective_date": current.effective_date,
            },
            "future_versions": [
                {"article_key": item.article_key, "effective_date": item.effective_date}
                for item in coverage.future_versions
            ],
            "unknown_date_versions": [
                item.article_key for item in coverage.unknown_date_versions
            ],
        })
        if not coverage.has_current_article:
            missing_articles.append(f"{law_id}:{article}")

    print(
        json.dumps(
            {
                "format": "graphrag-drift-temporal-coverage-v1",
                "as_of_date": as_of,
                "results": results,
                "missing_current_law_ids": missing,
                "articles": article_results,
                "missing_current_articles": missing_articles,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.require_current and (missing or missing_articles):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
