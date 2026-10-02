from __future__ import annotations

from typing import Literal, TypedDict

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
    depth: int
    followups: list[str]
    evidence: list[SearchHit]
    partials: list[tuple[str, float]]
    answer: str


def build_workflow(retriever: Retriever, reasoner: Reasoner | None = None):
    engine = GraphRAGEngine(retriever, reasoner)

    def route(state: GraphRAGState) -> str:
        return state["mode"]

    def local_search(state: GraphRAGState) -> GraphRAGState:
        return engine.local(state["query"], top_k=state.get("top_k", 5))

    def global_map_reduce(state: GraphRAGState) -> GraphRAGState:
        return engine.global_search(state["query"], top_k=state.get("top_k", 5))

    def drift_primer(state: GraphRAGState) -> GraphRAGState:
        return engine.drift_primer(state["query"], top_k=state.get("top_k", 5))

    def drift_expand(state: GraphRAGState) -> GraphRAGState:
        return engine.drift_expand(
            state["query"],
            evidence=state.get("evidence", []),
            followups=state.get("followups", []),
            depth=state.get("depth", 0),
            top_k=state.get("top_k", 5),
        )

    def drift_should_continue(state: GraphRAGState) -> str:
        if state.get("depth", 0) >= state.get("max_depth", 2):
            return "reduce"
        if not state.get("followups"):
            return "reduce"
        return "expand"

    def drift_reduce(state: GraphRAGState) -> GraphRAGState:
        return {
            "answer": engine.drift_reduce(
                state["query"],
                state.get("evidence", []),
            )
        }

    graph = StateGraph(GraphRAGState)
    graph.add_node("local", local_search)
    graph.add_node("global", global_map_reduce)
    graph.add_node("drift_primer", drift_primer)
    graph.add_node("drift_expand", drift_expand)
    graph.add_node("drift_reduce", drift_reduce)

    graph.add_conditional_edges(
        START,
        route,
        {
            "local": "local",
            "global": "global",
            "drift": "drift_primer",
        },
    )

    graph.add_edge("local", END)
    graph.add_edge("global", END)
    graph.add_conditional_edges(
        "drift_primer",
        drift_should_continue,
        {
            "expand": "drift_expand",
            "reduce": "drift_reduce",
        },
    )
    graph.add_conditional_edges(
        "drift_expand",
        drift_should_continue,
        {
            "expand": "drift_expand",
            "reduce": "drift_reduce",
        },
    )
    graph.add_edge("drift_reduce", END)

    return graph.compile()
