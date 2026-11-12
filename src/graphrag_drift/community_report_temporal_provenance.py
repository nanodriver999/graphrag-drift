from __future__ import annotations

import re
from typing import Any, Iterable

from .temporal_coverage import normalize_date


_ARTICLE_KEY = re.compile(
    r"^kr:law:(?P<law_id>[0-9]+):.*:effective:"
    r"(?P<effective_date>[0-9]{8}):art:(?P<article>[0-9]+(?:-[0-9]+)?)$"
)


def collect_community_source_provenance(
    overview: dict[str, Any],
    evidence_parts: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Derive auditable source identities without consulting an LLM.

    This tracks *input context*, not which claims the generated summary
    actually uses or whether the generated summary is legally correct.
    """
    by_key: dict[str, dict[str, str]] = {}
    chunk_count = 0
    undated_chunk_count = 0
    other_evidence_count = 0
    for part in evidence_parts:
        raw_evidence = part.get("evidence", part.get("evidence_nodes", []))
        if not isinstance(raw_evidence, list):
            raise ValueError("Community report evidence must be an array")
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            labels = item.get("labels", [])
            if not isinstance(labels, list) or "Chunk" not in labels:
                other_evidence_count += 1
                continue
            chunk_count += 1
            properties = item.get("properties", {})
            if not isinstance(properties, dict):
                undated_chunk_count += 1
                continue
            article_key = properties.get("articleKey")
            match = _ARTICLE_KEY.fullmatch(str(article_key or ""))
            effective = str(properties.get("officialEffectiveDate") or "")
            if not match or match["effective_date"] != effective:
                undated_chunk_count += 1
                continue
            by_key[match.string] = {
                "article_key": match.string,
                "law_id": match["law_id"],
                "article": match["article"],
                "effective_date": effective,
            }

    entities = overview.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("Community report overview.entities must be an array")
    return {
        "format": "graphrag-drift-community-source-provenance-v1",
        "source_article_versions": [by_key[key] for key in sorted(by_key)],
        "source_chunk_count": chunk_count,
        "undated_or_unversioned_chunk_count": undated_chunk_count,
        "other_evidence_count": other_evidence_count,
        "entity_description_count": len(entities),
        "scope": (
            "Input source context only. This metadata does not prove that "
            "every generated statement is supported by a dated legal source."
        ),
    }


def evaluate_community_source_date(
    provenance: dict[str, Any] | None,
    *,
    as_of_date: str,
) -> dict[str, Any]:
    as_of = normalize_date(as_of_date)
    if provenance is None:
        return {
            "status": "source_provenance_not_recorded",
            "as_of_date": as_of,
            "future_article_keys": [],
            "dated_article_count": None,
            "uncertain_source_count": None,
        }
    if provenance.get("format") != "graphrag-drift-community-source-provenance-v1":
        raise ValueError("Unsupported community source provenance format")
    rows = provenance.get("source_article_versions", [])
    if not isinstance(rows, list):
        raise ValueError("source_article_versions must be a list")
    future_keys: list[str] = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Invalid source article entry")
        effective = str(item.get("effective_date") or "")
        key = str(item.get("article_key") or "")
        if not _ARTICLE_KEY.fullmatch(key) or not re.fullmatch(r"[0-9]{8}", effective):
            raise ValueError("Invalid dated source article")
        if effective > as_of:
            future_keys.append(key)
    uncertain = (
        int(provenance.get("undated_or_unversioned_chunk_count", 0))
        + int(provenance.get("entity_description_count", 0))
        + int(provenance.get("other_evidence_count", 0))
    )
    return {
        "status": (
            "future_versions_in_report_input"
            if future_keys else "no_future_article_detected_in_report_input"
        ),
        "as_of_date": as_of,
        "future_article_keys": sorted(future_keys),
        "dated_article_count": len(rows),
        "uncertain_source_count": uncertain,
        "warning": (
            "A report generated from source context mixing date-effective "
            "versions cannot be certified as-of a past date without rebuilding."
            if future_keys else
            "No future source article found does not certify the generated "
            "summary; undated input and synthesized claims may still be stale."
        ),
    }
