from __future__ import annotations

from typing import Protocol

from .models import CommunityReport, SearchHit


class Reasoner(Protocol):
    def answer_local(self, query: str, hits: list[SearchHit]) -> str:
        ...

    def map_community(self, query: str, report: CommunityReport) -> tuple[str, float]:
        ...

    def reduce_global(self, query: str, partials: list[tuple[str, float]]) -> str:
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
