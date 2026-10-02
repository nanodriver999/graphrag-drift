from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import CommunityReport, SearchHit
from graphrag_drift.retrieval import InMemoryRetriever
from graphrag_drift.workflow import build_workflow


def make_app():
    retriever = InMemoryRetriever(
        hits=[
            SearchHit("e1", "GraphRAG uses graph entities and relationships", 0.9),
            SearchHit("e2", "DRIFT performs iterative local exploration", 0.8),
            SearchHit("e3", "Global search summarizes community reports", 0.7),
        ],
        reports=[
            CommunityReport("c1", "GraphRAG global search uses community summaries", 0.8),
            CommunityReport("c2", "DRIFT combines global context with local exploration", 0.95),
        ],
    )
    return build_workflow(retriever, DeterministicReasoner())


def test_local_search():
    app = make_app()
    out = app.invoke({"query": "GraphRAG entities", "mode": "local", "top_k": 2})
    assert out["answer"].startswith("LOCAL[GraphRAG entities]:")
    assert len(out["evidence"]) == 2


def test_global_search():
    app = make_app()
    out = app.invoke({"query": "DRIFT global", "mode": "global", "top_k": 2})
    assert out["answer"].startswith("GLOBAL[DRIFT global]:")
    assert out["partials"][0][1] in {0.8, 0.95}


def test_drift_search_loops_and_reduces():
    app = make_app()
    out = app.invoke(
        {
            "query": "DRIFT",
            "mode": "drift",
            "top_k": 1,
            "max_depth": 2,
        }
    )
    assert out["answer"].startswith("DRIFT[DRIFT]:")
    assert out["depth"] == 2
    assert len(out["evidence"]) >= 1
