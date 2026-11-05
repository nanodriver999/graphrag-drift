from __future__ import annotations

from threading import Barrier, get_ident

from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import CommunityReport
from graphrag_drift.retrieval import InMemoryRetriever


class ParallelGlobalReasoner(DeterministicReasoner):
    def __init__(self) -> None:
        self.barrier = Barrier(2, timeout=2)
        self.thread_ids: set[int] = set()

    def map_community(
        self,
        query: str,
        report: CommunityReport,
    ) -> tuple[str, float]:
        self.thread_ids.add(get_ident())
        self.barrier.wait()
        return (f"{report.id}:{report.summary}", report.score)


def retriever() -> InMemoryRetriever:
    return InMemoryRetriever(
        reports=[
            CommunityReport("c1", "one", 0.9),
            CommunityReport("c2", "two", 0.8),
        ]
    )


def test_global_search_executes_map_calls_concurrently() -> None:
    reasoner = ParallelGlobalReasoner()
    engine = GraphRAGEngine(retriever(), reasoner)

    result = engine.global_search(
        "query",
        top_k=2,
        map_concurrency=2,
    )

    assert result["map_concurrency"] == 2
    assert len(reasoner.thread_ids) == 2
    assert [text for text, _score in result["partials"]] == [
        "c1:one",
        "c2:two",
    ]


def test_global_search_can_preserve_sequential_mode() -> None:
    engine = GraphRAGEngine(retriever(), DeterministicReasoner())

    result = engine.global_search(
        "query",
        top_k=2,
        map_concurrency=1,
    )

    assert result["map_concurrency"] == 1
    assert len(result["partials"]) == 2
