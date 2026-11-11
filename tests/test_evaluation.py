from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import sleep

from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.evaluation import GraphRAGModeBenchmark, InstrumentedTextGenerator
from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import CommunityReport, SearchHit
import pytest

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
    assert rows["global"]["map_concurrency"] == 4
    assert rows["global"]["reduce_char_budget"] == 24000
    assert rows["global"]["reduce_partial_count"] == 2
    assert rows["global"]["reduce_char_count"] > 0

    assert rows["drift"]["llm_calls"] == 3
    assert rows["drift"]["depth"] == 1
    assert rows["drift"]["evidence_count"] == 4
    assert rows["drift"]["unique_evidence_count"] == 4
    assert rows["drift"]["primer_source"] == "community"
    assert rows["drift"]["primer_score"] == 90
    assert rows["drift"]["primer_report_count"] == 1
    assert rows["drift"]["primer_fold_count"] == 1
    assert rows["drift"]["action_concurrency"] == 4
    assert rows["drift"]["primer_concurrency"] == 5
    assert rows["drift"]["action_count"] == 2
    assert rows["drift"]["completed_action_count"] == 2
    assert rows["drift"]["frontier_count"] == 0
    assert rows["drift"]["epochs"] == 1


def test_mode_benchmark_records_named_drift_profile_settings() -> None:
    generator = InstrumentedTextGenerator(
        FakeGenerator(
            responses=[
                "local answer",
                "global partial",
                "global answer",
                '{"intermediate_answer":"primer","score":80,"follow_up_queries":[]}',
                "drift answer",
            ]
        )
    )
    reasoner = GraphRAGLLMReasoner(generator=generator, max_followups=20)
    engine = GraphRAGEngine(retriever=FakeRetriever(), reasoner=reasoner)
    benchmark = GraphRAGModeBenchmark(
        local_engine=engine,
        global_engine=engine,
        drift_engine=engine,
        generator_probe=generator,
    )

    result = benchmark.run(
        "test query",
        local_top_k=1,
        global_top_k=1,
        global_map_concurrency=8,
        global_reduce_char_budget=12000,
        drift_top_k=1,
        drift_profile="microsoft-like",
        drift_max_followups=20,
        drift_max_depth=3,
        drift_max_actions_per_depth=8,
        drift_community_top_k=2,
        drift_action_concurrency=32,
        drift_primer_folds=5,
        drift_primer_concurrency=32,
    )

    settings = result["settings"]
    assert settings["global_map_concurrency"] == 8
    assert settings["global_reduce_char_budget"] == 12000
    assert settings["drift_profile"] == "microsoft-like"
    assert settings["drift_max_followups"] == 20
    assert settings["drift_max_depth"] == 3
    assert settings["drift_action_concurrency"] == 32



class BlockingGenerator:
    def __init__(self) -> None:
        self.lock = Lock()
        self.active = 0
        self.max_active = 0

    def generate(self, prompt: str) -> str:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            sleep(0.05)
            return prompt
        finally:
            with self.lock:
                self.active -= 1


def test_instrumented_generator_limits_provider_inflight_requests() -> None:
    base = BlockingGenerator()
    generator = InstrumentedTextGenerator(
        base,
        max_concurrent_requests=1,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(generator.generate, ["a", "b", "c", "d"]))

    assert results == ["a", "b", "c", "d"]
    assert base.max_active == 1
    assert generator.call_count == 4


def test_instrumented_generator_rejects_invalid_provider_limit() -> None:
    with pytest.raises(ValueError, match="max_concurrent_requests"):
        InstrumentedTextGenerator(
            FakeGenerator(responses=[]),
            max_concurrent_requests=0,
        )


def test_drift_only_benchmark_accepts_custom_resumable_executor() -> None:
    generator = InstrumentedTextGenerator(FakeGenerator(responses=[]))
    engine = GraphRAGEngine(retriever=FakeRetriever(), reasoner=DeterministicReasoner())
    benchmark = GraphRAGModeBenchmark(
        local_engine=engine,
        global_engine=engine,
        drift_engine=engine,
        generator_probe=generator,
    )

    result = benchmark.run_drift_only(
        "test query",
        drift_top_k=1,
        drift_executor=lambda: {
            "status": "completed",
            "checkpoint_reused": True,
            "evidence": [SearchHit("cached", "cached evidence", 1.0)],
            "answer": "cached answer",
            "depth": 1,
            "followups": [],
            "primer_source": "community",
            "primer_score": 80,
            "primer_report_ids": ["c1"],
            "primer_fold_count": 1,
            "action_concurrency": 1,
            "primer_concurrency": 1,
            "action_count": 1,
            "completed_action_count": 1,
            "frontier_count": 0,
            "epochs": 1,
        },
    )

    row = result["results"][0]
    assert row["answer"] == "cached answer"
    assert row["checkpoint_reused"] is True
    assert row["run_status"] == "completed"
    assert row["llm_calls"] == 0
