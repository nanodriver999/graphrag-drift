from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .benchmark_suite import BenchmarkCase, run_benchmark_suite
from .drift_profile_comparison import compare_drift_profiles
from .profile_progress import ProfileBenchmarkProgress


FORMAT = "graphrag-drift-partial-profile-report-v1"


def build_partial_profile_report(
    cases: Iterable[BenchmarkCase],
    progress: ProfileBenchmarkProgress,
    *,
    profiles: tuple[str, ...] = ("economy", "microsoft-like"),
) -> dict[str, Any]:
    """Compare only like-for-like completed cases from durable benchmark cache.

    Never invoke a model or silently compare non-overlapping question sets.
    Incomplete or failed results stay visible as missing, not zero-quality data.
    """
    suite = list(cases)
    if len({case.id for case in suite}) != len(suite):
        raise ValueError("Duplicate benchmark case IDs")
    if len({case.query for case in suite}) != len(suite):
        raise ValueError("Duplicate benchmark questions cannot be joined to cache")
    if len(profiles) != len(set(profiles)) or len(profiles) < 2:
        raise ValueError("At least two distinct profiles are required")

    by_query = {case.query: case for case in suite}
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in progress.entries.values():
        query = entry.get("query")
        settings = entry.get("settings")
        rows = entry.get("results")
        if not isinstance(query, str) or query not in by_query:
            continue
        if not isinstance(settings, dict) or not isinstance(rows, list) or not rows:
            continue
        profile = settings.get("drift_profile")
        if profile not in profiles:
            continue
        row = rows[0]
        if not isinstance(row, dict) or row.get("mode") != "drift":
            continue
        if row.get("run_status", "completed") != "completed":
            continue
        key = (query, profile)
        if key in cached:
            raise ValueError(
                f"Ambiguous cached measurements for query and profile: {key}"
            )
        cached[key] = entry

    completed: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for profile in profiles:
        completed[profile] = [
            case.id for case in suite if (case.query, profile) in cached
        ]
        missing[profile] = [
            case.id for case in suite if (case.query, profile) not in cached
        ]

    paired_cases = [
        case for case in suite
        if all((case.query, profile) in cached for profile in profiles)
    ]
    paired = None
    if paired_cases:
        runs = {
            profile: run_benchmark_suite(
                paired_cases,
                lambda query, name=profile: deepcopy(cached[query, name]),
            )
            for profile in profiles
        }
        paired = compare_drift_profiles(
            runs, baseline_profile=profiles[0]
        )

    return {
        "format": FORMAT,
        "benchmark_identity": progress.identity,
        "total_cases": len(suite),
        "paired_case_count": len(paired_cases),
        "paired_case_ids": [case.id for case in paired_cases],
        "profiles": [
            {
                "name": profile,
                "completed_count": len(completed[profile]),
                "completed_case_ids": completed[profile],
                "missing_case_ids": missing[profile],
            }
            for profile in profiles
        ],
        "case_matrix": [
            {
                "id": case.id,
                "query": case.query,
                "completed_profiles": [
                    profile for profile in profiles
                    if (case.query, profile) in cached
                ],
            }
            for case in suite
        ],
        "paired_comparison": paired,
        "caveat": (
            "Means and quality rates use only cases completed by all profiles. "
            "Missing or failed cases are not scored. A partial experiment is "
            "not evidence of overall profile superiority."
        ),
    }


def render_partial_profile_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Partial DRIFT benchmark report",
        "",
        f"Complete matched pairs: **{report['paired_case_count']}/{report['total_cases']}**",
        "",
        "| Profile | Completed | Missing |",
        "| --- | ---: | --- |",
    ]
    for profile in report["profiles"]:
        names = ", ".join(profile["missing_case_ids"]) or "none"
        lines.append(
            f"| {profile['name']} | {profile['completed_count']}/"
            f"{report['total_cases']} | {names} |"
        )

    lines.extend(["", "## Paired cases", ""])
    if report["paired_case_ids"]:
        for case_id in report["paired_case_ids"]:
            lines.append(f"- {case_id}")
        paired = report["paired_comparison"]
        for profile in paired["profiles"]:
            lines.extend(["", f"### {profile['profile']} (paired cases only)", ""])
            for name in (
                "wall_ms",
                "llm_calls",
                "topic_coverage",
                "direct_article_source_coverage",
                "unique_direct_article_source_count",
            ):
                value = profile["metrics"].get(name, {}).get("mean")
                lines.append(f"- {name}: {value if value is not None else 'not available'}")
    else:
        lines.append("No question has been completed by every profile.")
    lines.extend([
        "",
        "Direct article source coverage measures retrieval of effective original "
        "law article IDs, not the factual or legal correctness of an answer. "
        "Only gold-annotated cases with source provenance have a score.",
        "",
        report["caveat"],
        "",
    ])
    return "\n".join(lines)
