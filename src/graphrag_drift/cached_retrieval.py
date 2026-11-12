from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .community_report_store import FileCommunityReportStore
from .community_report_temporal_provenance import evaluate_community_source_date
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
    as_of_date: str | None = None
    temporal_policy: Literal["none", "exclude_future", "require_date_clear_input"] = "none"

    def __post_init__(self) -> None:
        if self.temporal_policy not in {"none", "exclude_future", "require_date_clear_input"}:
            raise ValueError("Unknown Community Report temporal policy")
        if self.temporal_policy != "none" and not self.as_of_date:
            raise ValueError("Temporal Community Report policy requires as_of_date")

    def _allowed_for_date(self, report: CommunityReport) -> bool:
        if self.temporal_policy == "none":
            return True
        assert self.as_of_date is not None
        evaluation = evaluate_community_source_date(
            report.source_provenance,
            as_of_date=self.as_of_date,
        )
        if evaluation["status"] == "future_versions_in_report_input":
            return False
        if self.temporal_policy == "require_date_clear_input":
            return (
                evaluation["status"] == "no_future_article_detected_in_report_input"
                and evaluation["uncertain_source_count"] == 0
                and evaluation["dated_article_count"] > 0
            )
        # exclude_future keeps unversioned legacy reports for compatibility.
        # This never certifies their legal content as correct/current.
        return True

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        return self.local_retriever.local_search(query, top_k=top_k)

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        reports = [
            report for report in self.report_store.load()
            if self._allowed_for_date(report)
        ]
        terms = {term.lower() for term in query.split() if term.strip()}

        def rank(report: CommunityReport) -> tuple[int, float, str]:
            text = report.summary.lower()
            overlap = sum(1 for term in terms if term in text)
            return (overlap, report.score, report.id)

        return sorted(reports, key=rank, reverse=True)[:top_k]
