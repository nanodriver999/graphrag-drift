from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import CommunityReport, SearchHit
from graphrag_drift.retrieval import InMemoryRetriever


def make_engine():
    return GraphRAGEngine(
        InMemoryRetriever(
            hits=[
                SearchHit("e1", "GraphRAG uses graph entities and relationships", 0.9),
                SearchHit("e2", "DRIFT performs iterative local exploration", 0.8),
                SearchHit("e3", "Global search summarizes community reports", 0.7),
            ],
            reports=[
                CommunityReport("c1", "GraphRAG global search uses community summaries", 0.8),
                CommunityReport("c2", "DRIFT combines global context with local exploration", 0.95),
            ],
        ),
        DeterministicReasoner(),
    )


def test_local_engine():
    out = make_engine().local("GraphRAG entities", top_k=2)
    assert out["answer"].startswith("LOCAL[GraphRAG entities]:")
    assert [hit.id for hit in out["evidence"]] == ["e1", "e2"]


def test_global_engine_ranks_reduce_by_score():
    out = make_engine().global_search("DRIFT global", top_k=2)
    assert out["answer"].startswith("GLOBAL[DRIFT global]:")
    assert "c2:" in out["answer"].split(" | ")[0]


def test_drift_engine_iterates_until_max_depth():
    out = make_engine().drift("DRIFT", top_k=1, max_depth=2)
    assert out["depth"] == 2
    assert out["followups"] == []
    assert out["answer"].startswith("DRIFT[DRIFT]:")
    assert len(out["evidence"]) == 3
