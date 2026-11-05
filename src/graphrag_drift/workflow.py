from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .core import GraphRAGEngine
from .llm import Reasoner
from .models import SearchHit
from .retrieval import Retriever


SearchMode = Literal["local", "global", "drift"]


class GraphRAGState(TypedDict, total=False):
    query: str
    mode: SearchMode
    top_k: int
    max_depth: int
    max_actions_per_depth: int
    community_top_k: int
    action_concurrency: int
    primer_folds: int
    primer_concurrency: int
    depth: int
    followups: list[str]
    evidence: list[SearchHit]
    partials: list[tuple[str, float]]
    primer_answer: str
    primer_score: int
    primer_report_ids: list[str]
    primer_fold_count: int
    primer_source: str
    actions: dict[str, Any]
    hierarchy: list[dict[str, Any]]
    action_count: int
    completed_action_count: int
    frontier_count: int
    epochs: int
    answer: str


def build_workflow(retriever: Retriever, reasoner: Reasoner | None = None):
    """Build one LangGraph entry point for Local, Global, and ranked DRIFT.

    DRIFT intentionally delegates to GraphRAGEngine.drift() so the direct engine
    and LangGraph execution paths share the same ranked-action implementation.
    The serialized action graph remains available in the returned state.
    """

    engine = GraphRAGEngine(retriever, reasoner)

    def route(state: GraphRAGState) -> str:
        return state["mode"]

    def local_search(state: GraphRAGState) -> GraphRAGState:
        return engine.local(state["query"], top_k=state.get("top_k", 5))

    def global_map_reduce(state: GraphRAGState) -> GraphRAGState:
        return engine.global_search(state["query"], top_k=state.get("top_k", 5))

    def drift_ranked_search(state: GraphRAGState) -> GraphRAGState:
        return engine.drift(
            state["query"],
            top_k=state.get("top_k", 5),
            max_depth=state.get("max_depth", 2),
            max_actions_per_depth=state.get("max_actions_per_depth", 2),
            community_top_k=state.get("community_top_k", 1),
            action_concurrency=state.get("action_concurrency", 4),
            primer_folds=state.get("primer_folds", 5),
            primer_concurrency=state.get("primer_concurrency", 5),
        )

    graph = StateGraph(GraphRAGState)
    graph.add_node("local", local_search)
    graph.add_node("global", global_map_reduce)
    graph.add_node("drift", drift_ranked_search)

    graph.add_conditional_edges(
        START,
        route,
        {
            "local": "local",
            "global": "global",
            "drift": "drift",
        },
    )

    graph.add_edge("local", END)
    graph.add_edge("global", END)
    graph.add_edge("drift", END)

    return graph.compile()
