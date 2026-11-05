from __future__ import annotations

import json
from pathlib import Path

from graphrag_drift.benchmark_suite import (
    BenchmarkCase,
    load_benchmark_cases,
    run_benchmark_suite,
    summarize_benchmark_runs,
)


def test_load_benchmark_cases_reads_jsonl_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "query": "question one",
                        "tags": ["local", "law"],
                        "expected_article": "53-3",
                    }
                ),
                json.dumps({"id": "q2", "query": "question two"}),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_benchmark_cases(path)

    assert [case.id for case in cases] == ["q1", "q2"]
    assert cases[0].tags == ("local", "law")
    assert cases[0].metadata["expected_article"] == "53-3"


def test_summarize_benchmark_runs_aggregates_modes() -> None:
    runs = [
        {
            "results": [
                {
                    "mode": "local",
                    "wall_ms": 10,
                    "llm_calls": 1,
                    "evidence_count": 2,
                    "unique_evidence_count": 2,
                    "answer": "a",
                },
                {
                    "mode": "drift",
                    "wall_ms": 30,
                    "llm_calls": 3,
                    "evidence_count": 5,
                    "unique_evidence_count": 5,
                    "depth": 2,
                    "action_count": 3,
                    "completed_action_count": 3,
                    "answer": "d",
                },
            ]
        },
        {
            "results": [
                {
                    "mode": "local",
                    "wall_ms": 20,
                    "llm_calls": 1,
                    "evidence_count": 3,
                    "unique_evidence_count": 3,
                    "answer": "b",
                },
                {
                    "mode": "drift",
                    "wall_ms": 50,
                    "llm_calls": 4,
                    "evidence_count": 7,
                    "unique_evidence_count": 6,
                    "depth": 2,
                    "action_count": 4,
                    "completed_action_count": 3,
                    "answer": "",
                },
            ]
        },
    ]

    summary = summarize_benchmark_runs(runs)

    assert summary["case_count"] == 2
    assert summary["modes"]["local"]["wall_ms"]["mean"] == 15.0
    assert summary["modes"]["drift"]["llm_calls"]["mean"] == 3.5
    assert summary["modes"]["drift"]["nonempty_answer_rate"] == 0.5


def test_run_benchmark_suite_preserves_case_identity() -> None:
    cases = [
        BenchmarkCase("q1", "one", ("a",), {"difficulty": "easy"}),
        BenchmarkCase("q2", "two"),
    ]

    def runner(query: str) -> dict:
        return {
            "query": query,
            "results": [
                {
                    "mode": "local",
                    "wall_ms": 1,
                    "llm_calls": 1,
                    "answer": query,
                }
            ],
        }

    result = run_benchmark_suite(cases, runner)

    assert [row["id"] for row in result["cases"]] == ["q1", "q2"]
    assert result["cases"][0]["metadata"]["difficulty"] == "easy"
    assert result["summary"]["case_count"] == 2


def test_run_benchmark_suite_adds_quality_metrics() -> None:
    cases = [
        BenchmarkCase(
            "q1",
            "question",
            metadata={"expected_topics": ["선임", "신고", "교육"]},
        )
    ]

    def runner(query: str) -> dict:
        return {
            "query": query,
            "results": [
                {
                    "mode": "local",
                    "wall_ms": 1,
                    "llm_calls": 1,
                    "answer": "선임 후 신고를 해야 한다.",
                },
                {
                    "mode": "drift",
                    "wall_ms": 2,
                    "llm_calls": 2,
                    "answer": "제공된 증거만으로는 확인할 수 없습니다.",
                },
            ],
        }

    result = run_benchmark_suite(cases, runner)
    local, drift = result["cases"][0]["benchmark"]["results"]

    assert local["matched_topics"] == ["선임", "신고"]
    assert local["topic_coverage"] == 0.667
    assert not local["insufficient_answer"]

    assert drift["topic_coverage"] == 0.0
    assert drift["insufficient_answer"]
    assert result["summary"]["modes"]["local"]["topic_coverage"]["mean"] == 0.667
    assert result["summary"]["modes"]["drift"]["insufficient_answer_rate"] == 1.0


def test_run_benchmark_suite_rejects_invalid_expected_topics() -> None:
    cases = [
        BenchmarkCase(
            "q1",
            "question",
            metadata={"expected_topics": "not-a-list"},
        )
    ]

    def runner(query: str) -> dict:
        return {"query": query, "results": []}

    import pytest

    with pytest.raises(ValueError, match="expected_topics"):
        run_benchmark_suite(cases, runner)
