from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .quality_eval import evaluate_answer_quality


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    query: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Benchmark line {line_number} must be a JSON object")

        case_id = payload.get("id")
        query = payload.get("query")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Benchmark line {line_number} is missing id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Benchmark line {line_number} is missing query")

        raw_tags = payload.get("tags", [])
        if not isinstance(raw_tags, list) or any(
            not isinstance(item, str) for item in raw_tags
        ):
            raise ValueError(f"Benchmark line {line_number} tags must be strings")

        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"id", "query", "tags"}
        }
        cases.append(
            BenchmarkCase(
                id=case_id.strip(),
                query=query.strip(),
                tags=tuple(item.strip() for item in raw_tags if item.strip()),
                metadata=metadata,
            )
        )
    return cases


_NUMERIC_METRICS = (
    "wall_ms",
    "llm_calls",
    "llm_time_ms",
    "evidence_count",
    "unique_evidence_count",
    "depth",
    "remaining_followups",
    "primer_score",
    "primer_report_count",
    "action_count",
    "completed_action_count",
    "frontier_count",
    "epochs",
    "topic_coverage",
)


def _summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def summarize_benchmark_runs(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    run_list = list(runs)
    for run in run_list:
        for row in run.get("results", []):
            mode = str(row.get("mode") or "")
            if mode:
                by_mode.setdefault(mode, []).append(row)

    aggregate: dict[str, Any] = {}
    for mode, rows in sorted(by_mode.items()):
        metric_summary: dict[str, Any] = {"cases": len(rows)}
        for metric in _NUMERIC_METRICS:
            values = [
                float(row[metric])
                for row in rows
                if isinstance(row.get(metric), (int, float))
            ]
            if values:
                metric_summary[metric] = _summarize_values(values)

        nonempty_answers = sum(
            1 for row in rows if str(row.get("answer") or "").strip()
        )
        metric_summary["nonempty_answer_rate"] = round(
            nonempty_answers / len(rows),
            3,
        )
        insufficient_answers = sum(
            1 for row in rows if bool(row.get("insufficient_answer"))
        )
        metric_summary["insufficient_answer_rate"] = round(
            insufficient_answers / len(rows),
            3,
        )
        aggregate[mode] = metric_summary

    return {
        "format": "graphrag-drift-benchmark-suite-summary-v1",
        "case_count": len(run_list),
        "modes": aggregate,
    }


def run_benchmark_suite(
    cases: Iterable[BenchmarkCase],
    runner: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    mode_runs: list[dict[str, Any]] = []

    for case in cases:
        result = runner(case.query)
        expected_topics = case.metadata.get("expected_topics", [])
        if not isinstance(expected_topics, list) or any(
            not isinstance(item, str) for item in expected_topics
        ):
            raise ValueError(
                f"Benchmark case {case.id} expected_topics must be a string list"
            )

        for row in result.get("results", []):
            quality = evaluate_answer_quality(
                str(row.get("answer") or ""),
                expected_topics,
            )
            row.update(quality)

        mode_runs.append(result)
        case_rows.append(
            {
                "id": case.id,
                "query": case.query,
                "tags": list(case.tags),
                "metadata": case.metadata,
                "benchmark": result,
            }
        )

    return {
        "format": "graphrag-drift-benchmark-suite-v1",
        "cases": case_rows,
        "summary": summarize_benchmark_runs(mode_runs),
    }
