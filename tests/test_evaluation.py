from __future__ import annotations

from dataclasses import dataclass

from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.evaluation import GraphRAGModeBenchmark, InstrumentedTextGenerator
from graphrag_drift.models import CommunityReport, SearchHit
from graphrag_drift.search_llm_reasoner import GraphRAGLLMReasoner


@dataclass
class FakeGenerator:
    responses: list[str]

    def generate(self, prompt: str) -> str:
        return self.responses.pop(0)


@dataclass
class FakeRetriever:
    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        return [
            SearchHit("e1", f"evidence for {query}", 1.0),
            SearchHit("e2", "second", 0.8),
        ][:top_k]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        return [
            CommunityReport("c1", "community report", 1.0),
            CommunityReport("c2", "second report", 0.8),
        ][:top_k]


def test_mode_benchmark_records_llm_calls_and_evidence() -> None:
    generator = InstrumentedTextGenerator(
        FakeGenerator(
            responses=[
                "local answer",
                "global partial 1",
                "global partial 2",
                "global answer",
                '{"intermediate_answer":"primer answer","score":90,"follow_up_queries":["followup"]}',
                "[]",
                "drift answer",
            ]
        )
    )
    reasoner = GraphRAGLLMReasoner(generator=generator)
    engine = GraphRAGEngine(retriever=FakeRetriever(), reasoner=reasoner)
    benchmark = GraphRAGModeBenchmark(
        local_engine=engine,
        global_engine=engine,
        drift_engine=engine,
        generator_probe=generator,
    )

    result = benchmark.run(
        "test query",
        local_top_k=2,
        global_top_k=2,
        drift_top_k=1,
        drift_max_depth=1,
    )

    rows = {row["mode"]: row for row in result["results"]}
    assert rows["local"]["llm_calls"] == 1
    assert rows["local"]["evidence_count"] == 2

    assert rows["global"]["llm_calls"] == 3
    assert rows["global"]["evidence_count"] == 2

    assert rows["drift"]["llm_calls"] == 3
    assert rows["drift"]["depth"] == 1
    assert rows["drift"]["evidence_count"] == 2
    assert rows["drift"]["unique_evidence_count"] == 2
