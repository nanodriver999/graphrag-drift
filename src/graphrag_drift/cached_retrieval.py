from __future__ import annotations

from dataclasses import dataclass

from .community_report_store import FileCommunityReportStore
from .models import CommunityReport, SearchHit
from .retrieval import Retriever


@dataclass
class CachedCommunityReportRetriever:
    """Compose an existing retriever with file-cached Community Reports.

    Local search is delegated unchanged to the wrapped retriever. Global-search
    community reports are loaded from the versioned report cache and ranked with
    the same deterministic token-overlap strategy used by InMemoryRetriever.
    """

    local_retriever: Retriever
    report_store: FileCommunityReportStore

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        return self.local_retriever.local_search(query, top_k=top_k)

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        reports = self.report_store.load()
        terms = {term.lower() for term in query.split() if term.strip()}

        def rank(report: CommunityReport) -> tuple[int, float, str]:
            text = report.summary.lower()
            overlap = sum(1 for term in terms if term in text)
            return (overlap, report.score, report.id)

        return sorted(reports, key=rank, reverse=True)[:top_k]
