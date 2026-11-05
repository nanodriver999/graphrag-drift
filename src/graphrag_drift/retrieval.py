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
    """Compose Local Search context from entity/relationship, text-unit, and community sources."""

    retriever: Retriever
    community_prop: float = 0.25
    text_unit_prop: float = 0.50
    candidate_multiplier: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.community_prop < 1.0:
            raise ValueError("community_prop must satisfy 0 <= community_prop < 1")
        if not 0.0 <= self.text_unit_prop < 1.0:
            raise ValueError("text_unit_prop must satisfy 0 <= text_unit_prop < 1")
        if self.community_prop + self.text_unit_prop > 1.0:
            raise ValueError("community_prop + text_unit_prop must not exceed 1")
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")

    def _allocation(self, top_k: int) -> dict[str, int]:
        if top_k <= 0:
            return {"entity": 0, "text_unit": 0, "community": 0}

        proportions = {
            "entity": 1.0 - self.community_prop - self.text_unit_prop,
            "text_unit": self.text_unit_prop,
            "community": self.community_prop,
        }
        counts = {
            name: int(top_k * proportion)
            for name, proportion in proportions.items()
        }
        remaining = top_k - sum(counts.values())
        order = sorted(
            proportions,
            key=lambda name: (
                -(top_k * proportions[name] - counts[name]),
                {"entity": 0, "community": 1, "text_unit": 2}[name],
            ),
        )
        for name in order[:remaining]:
            counts[name] += 1
        return counts

    @staticmethod
    def _is_text_unit(hit: SearchHit) -> bool:
        return hit.id.startswith("chunk")

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        allocation = self._allocation(top_k)
        if top_k <= 0:
            return []

        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        candidates = self.retriever.local_search(query, top_k=candidate_k)
        entity_hits = [hit for hit in candidates if not self._is_text_unit(hit)]
        text_hits = [hit for hit in candidates if self._is_text_unit(hit)]

        reports = self.retriever.community_reports(
            query,
            top_k=allocation["community"],
        ) if allocation["community"] else []
        community_hits = [
            SearchHit(
                id=f"community:{report.id}",
                text=report.summary,
                score=report.score,
            )
            for report in reports
        ]

        selected = [
            *entity_hits[: allocation["entity"]],
            *text_hits[: allocation["text_unit"]],
        ]
        selected_ids = {hit.id for hit in selected}

        local_target = max(0, top_k - len(community_hits))
        if len(selected) < local_target:
            for hit in candidates:
                if hit.id in selected_ids:
                    continue
                selected.append(hit)
                selected_ids.add(hit.id)
                if len(selected) >= local_target:
                    break

        return [*selected[:local_target], *community_hits][:top_k]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        return self.retriever.community_reports(query, top_k=top_k)
