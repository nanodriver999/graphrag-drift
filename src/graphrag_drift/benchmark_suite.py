from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .quality_eval import evaluate_answer_quality
from .source_grounding import evaluate_article_source_support


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
    "retrieved_evidence_count",
    "unique_retrieved_evidence_count",
    "synthesized_context_count",
    "unique_direct_article_source_count",
    "direct_article_source_coverage",
    "depth",
    "remaining_followups",
    "primer_score",
    "primer_report_count",
    "primer_fold_count",
    "action_concurrency",
    "primer_concurrency",
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
        tracked = [
            row for row in rows
            if isinstance(row.get("successful_model_calls"), dict)
        ]
        if tracked:
            model_counts: dict[str, int] = {}
            for row in tracked:
                for model, count in row["successful_model_calls"].items():
                    if isinstance(count, int) and not isinstance(count, bool):
                        model_counts[str(model)] = model_counts.get(str(model), 0) + count
            metric_summary["successful_model_calls"] = model_counts
            metric_summary["model_usage_case_count"] = len(tracked)
            metric_summary["fallback_case_rate"] = round(
                sum(1 for row in tracked if row.get("used_fallback_model"))
                / len(tracked),
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

        expected_articles = case.metadata.get("expected_articles", [])
        if not isinstance(expected_articles, list) or any(
            not isinstance(item, str) for item in expected_articles
        ):
            raise ValueError(
                f"Benchmark case {case.id} expected_articles must be a string list"
            )
        as_of = case.metadata.get("as_of_date") or os.getenv("GRAPHRAG_AS_OF_DATE")

        for row in result.get("results", []):
            answer = str(row.get("answer") or "")
            quality = evaluate_answer_quality(answer, expected_topics)
            row.update(quality)
            if expected_articles:
                source_articles = row.get("direct_article_sources")
                if not isinstance(source_articles, list):
                    # Never call uninstrumented older runs zero-source matches.
                    row["article_source_audit_status"] = "source_provenance_not_recorded"
                elif not as_of:
                    row["article_source_audit_status"] = "as_of_date_not_set"
                else:
                    row.update(evaluate_article_source_support(
                        answer,
                        source_articles,
                        expected_articles,
                        as_of_date=str(as_of),
                    ))
                    row["article_source_audit_status"] = "checked"

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


def select_benchmark_cases(
    cases: list[BenchmarkCase],
    case_ids: list[str],
) -> list[BenchmarkCase]:
    """Filter to requested cases in dataset order for resumable case runs."""
    if not case_ids:
        return cases
    requested = set(case_ids)
    unknown = sorted(requested - {case.id for case in cases})
    if unknown:
        raise ValueError(f"Unknown benchmark case ID(s): {', '.join(unknown)}")
    return [case for case in cases if case.id in requested]


