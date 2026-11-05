from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegressionThresholds:
    wall_ms_increase_ratio: float = 0.50
    llm_calls_increase: float = 1.0
    topic_coverage_drop: float = 0.10
    insufficient_rate_increase: float = 0.10


def _mean(mode: dict[str, Any], metric: str) -> float | None:
    value = mode.get(metric)
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, (int, float)):
            return float(mean)
    return None


def compare_benchmark_summaries(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    thresholds: RegressionThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RegressionThresholds()
    baseline_modes = baseline.get("modes", {})
    current_modes = current.get("modes", {})

    mode_results: dict[str, Any] = {}
    regressions: list[dict[str, Any]] = []

    for mode in sorted(set(baseline_modes) | set(current_modes)):
        base = baseline_modes.get(mode, {})
        cur = current_modes.get(mode, {})
        checks: list[dict[str, Any]] = []

        base_wall = _mean(base, "wall_ms")
        cur_wall = _mean(cur, "wall_ms")
        if base_wall is not None and cur_wall is not None:
            ratio = None if base_wall == 0 else (cur_wall - base_wall) / base_wall
            regressed = (
                ratio is not None
                and ratio > thresholds.wall_ms_increase_ratio + 1e-9
            )
            check = {
                "metric": "wall_ms",
                "baseline": base_wall,
                "current": cur_wall,
                "delta_ratio": None if ratio is None else round(ratio, 4),
                "threshold": thresholds.wall_ms_increase_ratio,
                "regressed": regressed,
            }
            checks.append(check)
            if regressed:
                regressions.append({"mode": mode, **check})

        base_calls = _mean(base, "llm_calls")
        cur_calls = _mean(cur, "llm_calls")
        if base_calls is not None and cur_calls is not None:
            delta = cur_calls - base_calls
            regressed = delta > thresholds.llm_calls_increase + 1e-9
            check = {
                "metric": "llm_calls",
                "baseline": base_calls,
                "current": cur_calls,
                "delta": round(delta, 4),
                "threshold": thresholds.llm_calls_increase,
                "regressed": regressed,
            }
            checks.append(check)
            if regressed:
                regressions.append({"mode": mode, **check})

        base_coverage = _mean(base, "topic_coverage")
        cur_coverage = _mean(cur, "topic_coverage")
        if base_coverage is not None and cur_coverage is not None:
            drop = base_coverage - cur_coverage
            regressed = drop > thresholds.topic_coverage_drop + 1e-9
            check = {
                "metric": "topic_coverage",
                "baseline": base_coverage,
                "current": cur_coverage,
                "drop": round(drop, 4),
                "threshold": thresholds.topic_coverage_drop,
                "regressed": regressed,
            }
            checks.append(check)
            if regressed:
                regressions.append({"mode": mode, **check})

        base_insufficient = base.get("insufficient_answer_rate")
        cur_insufficient = cur.get("insufficient_answer_rate")
        if isinstance(base_insufficient, (int, float)) and isinstance(
            cur_insufficient, (int, float)
        ):
            increase = float(cur_insufficient) - float(base_insufficient)
            regressed = increase > thresholds.insufficient_rate_increase + 1e-9
            check = {
                "metric": "insufficient_answer_rate",
                "baseline": float(base_insufficient),
                "current": float(cur_insufficient),
                "increase": round(increase, 4),
                "threshold": thresholds.insufficient_rate_increase,
                "regressed": regressed,
            }
            checks.append(check)
            if regressed:
                regressions.append({"mode": mode, **check})

        mode_results[mode] = {
            "status": "regressed" if any(item["regressed"] for item in checks) else "ok",
            "checks": checks,
        }

    return {
        "format": "graphrag-drift-benchmark-regression-v1",
        "status": "regressed" if regressions else "ok",
        "regression_count": len(regressions),
        "regressions": regressions,
        "modes": mode_results,
    }


def compare_benchmark_results(
    baseline_result: dict[str, Any],
    current_result: dict[str, Any],
    *,
    thresholds: RegressionThresholds | None = None,
) -> dict[str, Any]:
    baseline_summary = baseline_result.get("summary", baseline_result)
    current_summary = current_result.get("summary", current_result)
    if not isinstance(baseline_summary, dict) or not isinstance(current_summary, dict):
        raise ValueError("Benchmark baseline/current summaries must be JSON objects")
    return compare_benchmark_summaries(
        baseline_summary,
        current_summary,
        thresholds=thresholds,
    )
