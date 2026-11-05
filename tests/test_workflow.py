from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import CommunityReport, SearchHit
from graphrag_drift.retrieval import InMemoryRetriever
from graphrag_drift.workflow import build_workflow


def make_retriever():
    return InMemoryRetriever(
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


def make_app():
    return build_workflow(make_retriever(), DeterministicReasoner())


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


def test_drift_workflow_uses_same_ranked_engine_result_shape():
    retriever = make_retriever()
    reasoner = DeterministicReasoner()
    app = build_workflow(retriever, reasoner)
    engine = GraphRAGEngine(retriever, reasoner)

    inputs = {
        "query": "DRIFT",
        "mode": "drift",
        "top_k": 1,
        "max_depth": 2,
        "max_actions_per_depth": 1,
    }
    out = app.invoke(inputs)
    direct = engine.drift(
        "DRIFT",
        top_k=1,
        max_depth=2,
        max_actions_per_depth=1,
    )

    assert out["answer"] == direct["answer"]
    assert out["depth"] == direct["depth"]
    assert out["followups"] == direct["followups"]
    assert out["primer_source"] == "community"
    assert out["primer_report_ids"] == direct["primer_report_ids"]
    assert out["action_count"] == direct["action_count"]
    assert out["completed_action_count"] == direct["completed_action_count"]
    assert out["frontier_count"] == direct["frontier_count"]
    assert out["epochs"] == direct["epochs"]
    assert out["actions"] == direct["actions"]
    assert [hit.id for hit in out["evidence"]] == [
        hit.id for hit in direct["evidence"]
    ]
