from __future__ import annotations

from typing import Any, Iterable


DEFAULT_DRIFT_COMPARISON_METRICS = (
    "wall_ms",
    "llm_calls",
    "llm_time_ms",
    "evidence_count",
    "unique_evidence_count",
    "topic_coverage",
    "action_count",
    "completed_action_count",
    "epochs",
    "primer_fold_count",
)


def _mean(summary: dict[str, Any], metric: str) -> float | None:
    value = summary.get(metric)
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
        return float(value["mean"])
    return None


def _drift_summary(suite: dict[str, Any]) -> dict[str, Any]:
    summary = suite.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    modes = summary.get("modes", {})
    if not isinstance(modes, dict):
        return {}
    drift = modes.get("drift", {})
    return drift if isinstance(drift, dict) else {}


def compare_drift_profiles(
    profile_runs: dict[str, dict[str, Any]],
    *,
    baseline_profile: str = "economy",
    metrics: Iterable[str] = DEFAULT_DRIFT_COMPARISON_METRICS,
) -> dict[str, Any]:
    if baseline_profile not in profile_runs:
        raise ValueError(f"Baseline profile {baseline_profile!r} is missing")

    baseline_summary = _drift_summary(profile_runs[baseline_profile])
    rows: list[dict[str, Any]] = []

    for profile_name, suite in profile_runs.items():
        drift = _drift_summary(suite)
        metric_rows: dict[str, Any] = {}
        for metric in metrics:
            current = _mean(drift, metric)
            baseline = _mean(baseline_summary, metric)
            delta = (
                None
                if current is None or baseline is None
                else round(current - baseline, 3)
            )
            ratio = (
                None
                if current is None or baseline in (None, 0)
                else round(current / baseline, 3)
            )
            metric_rows[metric] = {
                "mean": current,
                "baseline_mean": baseline,
                "delta": delta,
                "ratio": ratio,
            }

        insufficient = drift.get("insufficient_answer_rate")
        baseline_insufficient = baseline_summary.get("insufficient_answer_rate")
        insufficient_delta = (
            round(float(insufficient) - float(baseline_insufficient), 3)
            if isinstance(insufficient, (int, float))
            and isinstance(baseline_insufficient, (int, float))
            else None
        )

        rows.append(
            {
                "profile": profile_name,
                "is_baseline": profile_name == baseline_profile,
                "case_count": suite.get("summary", {}).get("case_count", 0)
                if isinstance(suite.get("summary"), dict)
                else 0,
                "metrics": metric_rows,
                "insufficient_answer_rate": insufficient,
                "insufficient_answer_rate_delta": insufficient_delta,
                "fallback_case_rate": drift.get("fallback_case_rate"),
                "model_usage_case_count": drift.get("model_usage_case_count"),
                "successful_model_calls": drift.get("successful_model_calls"),
            }
        )

    return {
        "format": "graphrag-drift-profile-comparison-v1",
        "baseline_profile": baseline_profile,
        "profiles": rows,
        "runs": profile_runs,
    }


def render_drift_profile_comparison_markdown(result: dict[str, Any]) -> str:
    baseline = result.get("baseline_profile", "economy")
    rows = result.get("profiles", [])
    if not isinstance(rows, list):
        rows = []

    lines = [
        "# DRIFT Profile Comparison",
        "",
        f"Baseline: **{baseline}**",
        "",
        "| Profile | Wall ms | LLM calls | Topic coverage | Insufficient rate | Evidence | Actions | Epochs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        def mean(name: str) -> str:
            item = metrics.get(name, {})
            if not isinstance(item, dict):
                return "-"
            value = item.get("mean")
            return "-" if value is None else str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("profile", "")),
                    mean("wall_ms"),
                    mean("llm_calls"),
                    mean("topic_coverage"),
                    str(row.get("insufficient_answer_rate", "-")),
                    mean("evidence_count"),
                    mean("action_count"),
                    mean("epochs"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Delta from baseline", ""])

    for row in rows:
        if not isinstance(row, dict) or row.get("is_baseline"):
            continue
        profile = str(row.get("profile") or "")
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        lines.append(f"### {profile}")
        lines.append("")
        for metric in (
            "wall_ms",
            "llm_calls",
            "topic_coverage",
            "evidence_count",
            "action_count",
            "epochs",
        ):
            item = metrics.get(metric, {})
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {metric}: delta={item.get('delta')}, ratio={item.get('ratio')}"
            )
        lines.append(
            "- insufficient_answer_rate: "
            f"delta={row.get('insufficient_answer_rate_delta')}"
        )
        lines.append("")

    lines.extend(["", "## Actual model usage (successful replies)", ""])
    for row in rows:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "")
        counts = row.get("successful_model_calls")
        lines.append(f"### {profile}")
        lines.append("")
        if isinstance(counts, dict):
            lines.append(
                f"- tracked cases: {row.get('model_usage_case_count')}"
            )
            lines.append(
                f"- fallback case rate: {row.get('fallback_case_rate')}"
            )
            for model, count in sorted(counts.items()):
                lines.append(f"- {model}: {count} successful replies")
        else:
            lines.append("- Model usage not recorded for this historical run")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
