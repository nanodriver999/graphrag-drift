from __future__ import annotations

from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.evaluation import GraphRAGModeBenchmark, InstrumentedTextGenerator
from graphrag_drift.global_llm_reasoner import TextGenerator
from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import CommunityReport, SearchHit
from graphrag_drift.retrieval import InMemoryRetriever


class NoopGenerator(TextGenerator):
    def generate(self, prompt: str) -> str:
        return "noop"


def test_run_drift_only_returns_single_drift_row_and_settings() -> None:
    retriever = InMemoryRetriever(
        hits=[SearchHit("h1", "local evidence", 1.0)],
        reports=[CommunityReport("c1", "community report", 0.9)],
    )
    probe = InstrumentedTextGenerator(NoopGenerator())
    engine = GraphRAGEngine(retriever, DeterministicReasoner())
    benchmark = GraphRAGModeBenchmark(
        local_engine=engine,
        global_engine=engine,
        drift_engine=engine,
        generator_probe=probe,
    )

    result = benchmark.run_drift_only(
        "query",
        drift_top_k=1,
        drift_profile="economy",
        drift_max_followups=2,
        drift_max_depth=1,
        drift_max_actions_per_depth=1,
        drift_community_top_k=0,
        drift_action_concurrency=1,
        drift_primer_folds=1,
        drift_primer_concurrency=1,
    )

    assert [row["mode"] for row in result["results"]] == ["drift"]
    assert result["settings"]["drift_profile"] == "economy"
    assert result["settings"]["drift_max_depth"] == 1
