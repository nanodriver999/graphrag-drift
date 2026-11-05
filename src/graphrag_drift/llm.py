from __future__ import annotations

from typing import Protocol

from .models import CommunityReport, DriftActionResult, DriftPrimerResult, SearchHit


class Reasoner(Protocol):
    def answer_local(self, query: str, hits: list[SearchHit]) -> str:
        ...

    def map_community(self, query: str, report: CommunityReport) -> tuple[str, float]:
        ...

    def reduce_global(self, query: str, partials: list[tuple[str, float]]) -> str:
        ...

    def prime_drift(
        self,
        query: str,
        reports: list[CommunityReport],
    ) -> DriftPrimerResult:
        ...

    def answer_drift_action(
        self,
        global_query: str,
        action_query: str,
        hits: list[SearchHit],
        *,
        depth: int,
    ) -> DriftActionResult:
        ...

    def generate_followups(
        self,
        query: str,
        evidence: list[SearchHit],
        *,
        depth: int,
    ) -> list[str]:
        ...

    def reduce_drift(self, query: str, evidence: list[SearchHit]) -> str:
        ...


class DeterministicReasoner:
    """Small deterministic reasoner used for tests and local smoke runs.

    Replace this with an LLM-backed implementation in production.
    """

    def answer_local(self, query: str, hits: list[SearchHit]) -> str:
        joined = " | ".join(hit.text for hit in hits)
        return f"LOCAL[{query}]: {joined}"

    def map_community(self, query: str, report: CommunityReport) -> tuple[str, float]:
        return (f"{report.id}: {report.summary}", report.score)

    def reduce_global(self, query: str, partials: list[tuple[str, float]]) -> str:
        ranked = sorted(partials, key=lambda x: x[1], reverse=True)
        return f"GLOBAL[{query}]: " + " | ".join(text for text, _ in ranked)

    def prime_drift(
        self,
        query: str,
        reports: list[CommunityReport],
    ) -> DriftPrimerResult:
        if not reports:
            return DriftPrimerResult("", 0, [])

        ranked = sorted(reports, key=lambda item: item.score, reverse=True)
        joined = " | ".join(report.summary for report in ranked)
        return DriftPrimerResult(
            intermediate_answer=f"DRIFT_PRIMER[{query}]: {joined}",
            score=100,
            follow_up_queries=[f"{query} followup 1"],
        )

    def answer_drift_action(
        self,
        global_query: str,
        action_query: str,
        hits: list[SearchHit],
        *,
        depth: int,
    ) -> DriftActionResult:
        joined = " | ".join(hit.text for hit in hits)
        score = max((hit.score for hit in hits), default=0.0)
        followups = [] if depth >= 2 else [f"{action_query} detail {depth + 1}"]
        return DriftActionResult(
            response=f"DRIFT_ACTION[{action_query}]: {joined}",
            score=max(0, min(100, round(score * 100))),
            follow_up_queries=followups,
        )

    def generate_followups(
        self,
        query: str,
        evidence: list[SearchHit],
        *,
        depth: int,
    ) -> list[str]:
        if depth >= 2:
            return []
        return [f"{query} followup {depth + 1}"]

    def reduce_drift(self, query: str, evidence: list[SearchHit]) -> str:
        unique: dict[str, SearchHit] = {hit.id: hit for hit in evidence}
        joined = " | ".join(hit.text for hit in unique.values())
        return f"DRIFT[{query}]: {joined}"
