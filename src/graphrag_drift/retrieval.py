from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import CommunityReport, SearchHit


class Retriever(Protocol):
    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        ...

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        ...


@dataclass
class InMemoryRetriever:
    hits: list[SearchHit]
    reports: list[CommunityReport]

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        terms = {t.lower() for t in query.split() if t.strip()}

        def rank(hit: SearchHit) -> tuple[int, float]:
            text = hit.text.lower()
            overlap = sum(1 for term in terms if term in text)
            return (overlap, hit.score)

        return sorted(self.hits, key=rank, reverse=True)[:top_k]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        terms = {t.lower() for t in query.split() if t.strip()}

        def rank(report: CommunityReport) -> tuple[int, float]:
            text = report.summary.lower()
            overlap = sum(1 for term in terms if term in text)
            return (overlap, report.score)

        return sorted(self.reports, key=rank, reverse=True)[:top_k]


@dataclass
class MixedLocalContextRetriever:
    """Reserve part of Local Search context for Community Reports.

    The wrapped retriever continues to own entity/relationship/text-unit ranking.
    This adapter only enforces a stable context quota so scores from different
    source families do not have to be calibrated against each other.
    """

    retriever: Retriever
    community_prop: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.community_prop < 1.0:
            raise ValueError("community_prop must satisfy 0 <= community_prop < 1")

    def _allocation(self, top_k: int) -> tuple[int, int]:
        if top_k <= 0:
            return (0, 0)
        if self.community_prop <= 0.0 or top_k == 1:
            return (top_k, 0)

        community_k = max(1, round(top_k * self.community_prop))
        community_k = min(community_k, top_k - 1)
        return (top_k - community_k, community_k)

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        local_k, community_k = self._allocation(top_k)
        hits = self.retriever.local_search(query, top_k=local_k)
        if community_k:
            reports = self.retriever.community_reports(query, top_k=community_k)
            hits.extend(
                SearchHit(
                    id=f"community:{report.id}",
                    text=report.summary,
                    score=report.score,
                )
                for report in reports
            )
        return hits[:top_k]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        return self.retriever.community_reports(query, top_k=top_k)
