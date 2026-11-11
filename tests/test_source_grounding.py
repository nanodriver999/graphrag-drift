from __future__ import annotations

import pytest

from graphrag_drift.source_grounding import (
    collect_article_sources,
    evaluate_article_source_support,
    parse_article_source_id,
)


CURRENT = "chunk-article:kr:law:011435:mst:283000:effective:20260520:art:53-3"
FUTURE = "chunk-article:kr:law:011435:mst:283347:effective:20270220:art:53-3"
OTHER_LAW = "chunk-article:kr:law:011484:mst:283000:effective:20260520:art:53-3"


def test_article_source_rejects_synthetic_and_unversioned_results() -> None:
    assert parse_article_source_id(CURRENT) == {
        "law_id": "011435",
        "article": "53-3",
        "effective_date": "20260520",
        "source_id": CURRENT,
    }
    assert parse_article_source_id("drift-action:제53조의3") is None
    assert parse_article_source_id("entity:kr:law:011435") is None
    assert parse_article_source_id("chunk:kr:law:011435") is None
    assert parse_article_source_id("chunk-article:kr:law:011435:art:53-3") is None


def test_article_sources_deduplicate_versioned_chunk_ids() -> None:
    sources = collect_article_sources([
        CURRENT, CURRENT, "drift-primer", "drift-action:hello", FUTURE,
    ])
    assert len(sources) == 2
    assert {row["effective_date"] for row in sources} == {"20260520", "20270220"}


def test_current_dated_source_required_for_gold_article_match() -> None:
    row = evaluate_article_source_support(
        "법 제53조의3에 따른 안전관리규정 검토",
        collect_article_sources([CURRENT, FUTURE]),
        ["011435:53-3"],
        as_of_date="2026-10-05",
    )
    assert row["direct_article_source_coverage"] == 1.0
    assert row["matched_expected_article_sources"] == ["011435:53-3"]
    assert row["expected_article_mentions_without_direct_source"] == []


def test_future_only_article_cannot_support_present_answer_claim() -> None:
    row = evaluate_article_source_support(
        "원자력안전법 제53조의3을 준수합니다.",
        collect_article_sources([FUTURE, OTHER_LAW]),
        ["011435:53-3"],
        as_of_date="2026-10-05",
    )
    assert row["direct_article_source_coverage"] == 0.0
    assert row["future_only_expected_article_sources"] == ["011435:53-3"]
    assert row["expected_article_mentions_without_direct_source"] == ["011435:53-3"]


def test_no_direct_source_does_not_imply_answer_is_factually_false() -> None:
    row = evaluate_article_source_support(
        "원자력안전법 제53조의3은 이런 업무를 규정한다.",
        [],
        ["011435:53-3"],
        as_of_date="2026-10-05",
    )
    assert row["missing_expected_article_sources"] == ["011435:53-3"]
    assert row["expected_article_mentions_without_direct_source"] == ["011435:53-3"]


def test_article_53_and_article_53_3_are_not_equated() -> None:
    row = evaluate_article_source_support(
        "법 제53조 제3항 규정",
        [],
        ["011435:53-3"],
        as_of_date="2026-10-05",
    )
    assert row["expected_article_mentions_without_direct_source"] == []


def test_reject_invalid_gold_article_syntax() -> None:
    with pytest.raises(ValueError, match="Expected article"):
        evaluate_article_source_support(
            "answer", [], ["원자력안전법 제53조의3"], as_of_date="2026-10-05"
        )
