from graphrag_drift.temporal_coverage import LawVersion, evaluate_law_coverage


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
