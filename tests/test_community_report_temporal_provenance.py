from __future__ import annotations

import pytest

from graphrag_drift.community_report_temporal_provenance import (
    collect_community_source_provenance,
    evaluate_community_source_date,
)


def _part() -> dict:
    return {
        "evidence": [
            {
                "labels": ["Chunk"],
                "properties": {
                    "articleKey": "kr:law:011435:mst:1:effective:20260520:art:53-3",
                    "officialEffectiveDate": "20260520",
                    "lawId": "011435",
                },
            },
            {
                "labels": ["Chunk"],
                "properties": {
                    "articleKey": "kr:law:011435:mst:2:effective:20270220:art:53-3",
                    "officialEffectiveDate": "20270220",
                    "lawId": "011435",
                },
            },
            {
                "labels": ["Chunk"],
                "properties": {"text": "an undated source"},
            },
        ],
    }


def test_community_source_provenance_flags_future_and_unknown_inputs() -> None:
    metadata = collect_community_source_provenance(
        {"entities": [{"properties": {"name": "manager"}}]},
        [_part(), _part()],
    )
    assert metadata["source_chunk_count"] == 6
    assert metadata["undated_or_unversioned_chunk_count"] == 2
    assert metadata["entity_description_count"] == 1
    assert len(metadata["source_article_versions"]) == 2

    evaluation = evaluate_community_source_date(
        metadata, as_of_date="2026-10-05",
    )
    assert evaluation["status"] == "future_versions_in_report_input"
    assert len(evaluation["future_article_keys"]) == 1
    assert evaluation["dated_article_count"] == 2
    assert evaluation["uncertain_source_count"] == 3
    assert evaluation["future_article_keys"][0].endswith("20270220:art:53-3")


def test_community_source_provenance_with_only_current_articles_is_not_certified() -> None:
    part = _part()
    part["evidence"] = part["evidence"][:1]
    metadata = collect_community_source_provenance(
        {"entities": [{"properties": {"description": "synthetic"}}]},
        [part],
    )
    evaluation = evaluate_community_source_date(metadata, as_of_date="2026-10-05")
    assert evaluation["future_article_keys"] == []
    assert evaluation["status"] == "no_future_article_detected_in_report_input"
    assert "does not certify" in evaluation["warning"]
    assert evaluation["uncertain_source_count"] == 1


def test_legacy_cached_report_without_provenance_is_unknown_not_safe() -> None:
    result = evaluate_community_source_date(None, as_of_date="2026-10-05")
    assert result["status"] == "source_provenance_not_recorded"
    assert result["dated_article_count"] is None
    assert result["future_article_keys"] == []


def test_source_provenance_rejects_invalid_evidence_structure() -> None:
    with pytest.raises(ValueError, match="evidence"):
        collect_community_source_provenance(
            {"entities": []},
            [{"evidence": "not a list"}],
        )


def test_community_source_dates_must_have_consistent_versioned_article_keys() -> None:
    part = {"evidence": [
        {
            "labels": ["Chunk"],
            "properties": {
                "articleKey": "kr:law:011435:effective:20270220:art:53-3",
                "officialEffectiveDate": "20260520",
            },
        },
    ]}
    metadata = collect_community_source_provenance(
        {"entities": []}, [part]
    )
    assert metadata["source_article_versions"] == []
    assert metadata["undated_or_unversioned_chunk_count"] == 1
