from __future__ import annotations

from typing import Any


def _metric_mean(mode: dict[str, Any], key: str) -> str:
    value = mode.get(key)
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
        return str(value["mean"])
    return "-"


def render_benchmark_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary", {})
    modes = summary.get("modes", {}) if isinstance(summary, dict) else {}

    lines = [
        "# GraphRAG Benchmark Summary",
        "",
        f"Cases: **{summary.get('case_count', 0)}**",
        "",
        "| Mode | Wall ms mean | LLM calls mean | Evidence mean | Topic coverage mean | Insufficient rate | Actions mean | Epochs mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for mode_name in ("local", "global", "drift"):
        mode = modes.get(mode_name, {})
        if not isinstance(mode, dict):
            mode = {}
        insufficient = mode.get("insufficient_answer_rate", "-")
        lines.append(
            "| "
            + " | ".join(
                [
                    mode_name,
                    _metric_mean(mode, "wall_ms"),
                    _metric_mean(mode, "llm_calls"),
                    _metric_mean(mode, "evidence_count"),
                    _metric_mean(mode, "topic_coverage"),
                    str(insufficient),
                    _metric_mean(mode, "action_count"),
                    _metric_mean(mode, "epochs"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Per-case answers", ""])

    for case in result.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "")
        query = str(case.get("query") or "")
        lines.extend([f"### {case_id}", "", f"**Query:** {query}", ""])
        benchmark = case.get("benchmark", {})
        rows = benchmark.get("results", []) if isinstance(benchmark, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            mode = str(row.get("mode") or "")
            answer = str(row.get("answer") or "").strip()
            coverage = row.get("topic_coverage", "-")
            insufficient = row.get("insufficient_answer", False)
            lines.extend(
                [
                    f"**{mode}** — topic coverage: {coverage}, insufficient: {insufficient}",
                    "",
                    answer or "_empty answer_",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"
