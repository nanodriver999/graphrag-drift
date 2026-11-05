from graphrag_drift.benchmark_regression import (
    RegressionThresholds,
    compare_benchmark_results,
)


def summary(
    *,
    local_wall: float = 10,
    local_calls: float = 1,
    local_coverage: float = 0.8,
    local_insufficient: float = 0.0,
) -> dict:
    return {
        "summary": {
            "modes": {
                "local": {
                    "wall_ms": {"mean": local_wall},
                    "llm_calls": {"mean": local_calls},
                    "topic_coverage": {"mean": local_coverage},
                    "insufficient_answer_rate": local_insufficient,
                }
            }
        }
    }


def test_regression_comparator_returns_ok_within_thresholds() -> None:
    result = compare_benchmark_results(
        summary(),
        summary(
            local_wall=14,
            local_calls=2,
            local_coverage=0.7,
            local_insufficient=0.1,
        ),
    )

    assert result["status"] == "ok"
    assert result["regression_count"] == 0


def test_regression_comparator_detects_quality_and_cost_regressions() -> None:
    result = compare_benchmark_results(
        summary(),
        summary(
            local_wall=20,
            local_calls=3,
            local_coverage=0.5,
            local_insufficient=0.3,
        ),
        thresholds=RegressionThresholds(
            wall_ms_increase_ratio=0.5,
            llm_calls_increase=1,
            topic_coverage_drop=0.1,
            insufficient_rate_increase=0.1,
        ),
    )

    assert result["status"] == "regressed"
    metrics = {row["metric"] for row in result["regressions"]}
    assert metrics == {
        "wall_ms",
        "llm_calls",
        "topic_coverage",
        "insufficient_answer_rate",
    }


def test_regression_comparator_handles_summary_objects_directly() -> None:
    baseline = summary()["summary"]
    current = summary(local_wall=12)["summary"]

    result = compare_benchmark_results(baseline, current)

    assert result["status"] == "ok"
