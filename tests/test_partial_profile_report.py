from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag_drift.benchmark_suite import BenchmarkCase
from graphrag_drift.partial_profile_report import (
    build_partial_profile_report,
    render_partial_profile_report_markdown,
)
from graphrag_drift.profile_progress import ProfileBenchmarkProgress


CASES = [
    BenchmarkCase("q1", "question one", metadata={"expected_topics": ["alpha"]}),
    BenchmarkCase("q2", "question two", metadata={"expected_topics": ["beta"]}),
    BenchmarkCase("q3", "question three"),
]


def _measurement(query: str, profile: str, answer: str, wall_ms: float) -> dict:
    return {
        "format": "graphrag-drift-mode-benchmark-v1",
        "query": query,
        "settings": {"drift_profile": profile},
        "results": [{
            "mode": "drift",
            "answer": answer,
            "wall_ms": wall_ms,
            "llm_calls": 4,
            "run_status": "completed",
        }],
    }


def test_partial_report_only_compares_matched_case_ids(tmp_path: Path) -> None:
    progress = ProfileBenchmarkProgress.load(tmp_path / "progress.json")
    progress.put("e1", _measurement("question one", "economy", "alpha", 10))
    progress.put("m1", _measurement("question one", "microsoft-like", "alpha", 30))
    progress.put("e2", _measurement("question two", "economy", "beta", 90))
    progress.put("m3", _measurement("question three", "microsoft-like", "unmatched", 40))

    report = build_partial_profile_report(CASES, progress)
    assert report["paired_case_ids"] == ["q1"]
    assert report["paired_case_count"] == 1
    assert report["profiles"][0]["completed_case_ids"] == ["q1", "q2"]
    assert report["profiles"][0]["missing_case_ids"] == ["q3"]
    assert report["profiles"][1]["completed_case_ids"] == ["q1", "q3"]
    assert report["profiles"][1]["missing_case_ids"] == ["q2"]

    paired_rows = {row["profile"]: row for row in report["paired_comparison"]["profiles"]}
    assert paired_rows["economy"]["metrics"]["wall_ms"]["mean"] == 10
    assert paired_rows["microsoft-like"]["metrics"]["wall_ms"]["mean"] == 30
    assert paired_rows["economy"]["metrics"]["topic_coverage"]["mean"] == 1
    assert "q2" in render_partial_profile_report_markdown(report)
    assert report["benchmark_identity"] is None


def test_partial_report_handles_no_matched_cases(tmp_path: Path) -> None:
    progress = ProfileBenchmarkProgress.load(tmp_path / "progress.json")
    progress.put("e2", _measurement("question two", "economy", "beta", 10))
    progress.put("m1", _measurement("question one", "microsoft-like", "alpha", 30))
    report = build_partial_profile_report(CASES, progress)
    assert report["paired_comparison"] is None
    assert report["paired_case_count"] == 0
    assert "No question has been completed" in render_partial_profile_report_markdown(report)


def test_partial_report_never_counts_failed_cache_entries(tmp_path: Path) -> None:
    progress = ProfileBenchmarkProgress.load(tmp_path / "progress.json")
    result = _measurement("question one", "economy", "alpha", 10)
    result["results"][0]["run_status"] = "failed"
    progress.put("e1", result)
    report = build_partial_profile_report(CASES, progress)
    assert report["profiles"][0]["completed_count"] == 0


def test_partial_report_rejects_multiple_results_for_same_pair(tmp_path: Path) -> None:
    progress = ProfileBenchmarkProgress.load(tmp_path / "progress.json")
    progress.put("e1", _measurement("question one", "economy", "alpha", 10))
    progress.put("different-key", _measurement("question one", "economy", "alpha", 20))
    with pytest.raises(ValueError, match="Ambiguous cached"):
        build_partial_profile_report(CASES, progress)


def test_partial_report_preserves_strict_identity(tmp_path: Path) -> None:
    identity = {"model": "beam-free", "app_revision": "fixed-revision"}
    progress = ProfileBenchmarkProgress.load(tmp_path / "tracked.json", identity=identity)
    progress.put("m1", _measurement("question one", "microsoft-like", "alpha", 50))
    reloaded = ProfileBenchmarkProgress.load(progress.path, identity=identity)
    report = build_partial_profile_report(CASES, reloaded)
    assert report["benchmark_identity"] == identity
