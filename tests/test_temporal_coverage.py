from graphrag_drift.temporal_coverage import (
    ArticleVersion,
    LawVersion,
    evaluate_article_coverage,
    evaluate_law_coverage,
)


def test_temporal_coverage_selects_latest_current_version() -> None:
    versions = [
        LawVersion("011435", "원자력안전법", "20250101", "old"),
        LawVersion("011435", "원자력안전법", "20260519", "current"),
        LawVersion("011435", "원자력안전법", "20270220", "future"),
    ]

    coverage = evaluate_law_coverage(
        "011435",
        versions,
        as_of_date="2026-10-05",
    )

    assert coverage.has_current_version
    assert coverage.latest_current is not None
    assert coverage.latest_current.version_key == "current"
    assert [item.version_key for item in coverage.future_versions] == ["future"]


def test_temporal_coverage_detects_future_only_law() -> None:
    versions = [
        LawVersion("011435", "원자력안전법", "20270220", "future"),
        LawVersion("011484", "원자력안전법 시행령", "20260701", "decree"),
    ]

    coverage = evaluate_law_coverage(
        "011435",
        versions,
        as_of_date="20261005",
    )

    assert not coverage.has_current_version
    assert coverage.latest_current is None
    assert [item.effective_date for item in coverage.future_versions] == ["20270220"]



def test_article_coverage_detects_future_only_required_article() -> None:
    versions = [
        ArticleVersion(
            "011435", "53-3", "20270220",
            "kr:law:011435:effective:20270220:art:53-3",
        ),
        ArticleVersion(
            "011435", "53-4", "20260501",
            "kr:law:011435:effective:20260501:art:53-4",
        ),
    ]
    result = evaluate_article_coverage(
        "011435", "53-3", versions, as_of_date="2026-10-05",
    )
    assert not result.has_current_article
    assert result.latest_current is None
    assert len(result.future_versions) == 1
    assert result.future_versions[0].effective_date == "20270220"


def test_article_coverage_requires_exact_article_not_just_law_version() -> None:
    versions = [
        ArticleVersion("011435", "53", "20260101", "law:art:53"),
        ArticleVersion("011435", "53-3", "20260520", "law:art:53-3"),
        ArticleVersion("011435", "53-3", "20270220", "new:art:53-3"),
        ArticleVersion("011484", "53-3", "20260101", "decree:art:53-3"),
    ]
    result = evaluate_article_coverage(
        "011435", "53-3", versions, as_of_date="20261005",
    )
    assert result.has_current_article
    assert result.latest_current.article_key == "law:art:53-3"
    assert len(result.future_versions) == 1


def test_article_coverage_does_not_treat_undated_text_as_current() -> None:
    versions = [
        ArticleVersion("011435", "53-3", "", "undated:art:53-3"),
    ]
    result = evaluate_article_coverage(
        "011435", "53-3", versions, as_of_date="2026-10-05",
    )
    assert not result.has_current_article
    assert len(result.unknown_date_versions) == 1
