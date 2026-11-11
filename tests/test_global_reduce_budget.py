from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import CommunityReport, SearchHit


@dataclass
class FixedRetriever:
    reports: list[CommunityReport]

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        return []

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        return self.reports[:top_k]


@dataclass
class RecordingReasoner(DeterministicReasoner):
    reduced_partials: list[tuple[str, float]] = field(default_factory=list)

    def map_community(
        self,
        query: str,
        report: CommunityReport,
    ) -> tuple[str, float]:
        return (report.summary, report.score)

    def reduce_global(
        self,
        query: str,
        partials: list[tuple[str, float]],
    ) -> str:
        self.reduced_partials = list(partials)
        return "reduced"


def test_global_reduce_budget_keeps_highest_score_partials_first() -> None:
    reasoner = RecordingReasoner()
    engine = GraphRAGEngine(
        FixedRetriever(
            [
                CommunityReport("low", "LLLL", 0.4),
                CommunityReport("high", "HHHHHH", 0.9),
                CommunityReport("mid", "MMMM", 0.7),
            ]
        ),
        reasoner,
    )

    result = engine.global_search(
        "query",
        top_k=3,
        map_concurrency=1,
        reduce_char_budget=10,
    )

    assert reasoner.reduced_partials == [
        ("HHHHHH", 0.9),
        ("MMMM", 0.7),
    ]
    assert result["reduce_partial_count"] == 2
    assert result["reduce_char_count"] == 10
    assert result["reduce_char_budget"] == 10


def test_global_reduce_budget_truncates_one_oversized_top_partial() -> None:
    reasoner = RecordingReasoner()
    engine = GraphRAGEngine(
        FixedRetriever([CommunityReport("c1", "ABCDEFGHIJ", 1.0)]),
        reasoner,
    )

    result = engine.global_search(
        "query",
        top_k=1,
        reduce_char_budget=5,
    )

    assert reasoner.reduced_partials == [("AB...", 1.0)]
    assert result["reduce_char_count"] == 5


def test_global_reduce_budget_rejects_non_positive_budget() -> None:
    engine = GraphRAGEngine(
        FixedRetriever([CommunityReport("c1", "text", 1.0)]),
        RecordingReasoner(),
    )

    with pytest.raises(ValueError, match="reduce_char_budget"):
        engine.global_search("query", reduce_char_budget=0)
