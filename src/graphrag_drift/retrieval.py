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
