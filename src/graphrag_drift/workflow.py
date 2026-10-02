from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import DeterministicReasoner, Reasoner
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
    reasoner = reasoner or DeterministicReasoner()

    def route(state: GraphRAGState) -> str:
        return state["mode"]

    def local_search(state: GraphRAGState) -> GraphRAGState:
        hits = retriever.local_search(
            state["query"],
            top_k=state.get("top_k", 5),
        )
        return {
            "evidence": hits,
            "answer": reasoner.answer_local(state["query"], hits),
        }

    def global_map_reduce(state: GraphRAGState) -> GraphRAGState:
        reports = retriever.community_reports(
            state["query"],
            top_k=state.get("top_k", 5),
        )
        partials = [reasoner.map_community(state["query"], report) for report in reports]
        return {
            "partials": partials,
            "answer": reasoner.reduce_global(state["query"], partials),
        }

    def drift_primer(state: GraphRAGState) -> GraphRAGState:
        hits = retriever.local_search(
            state["query"],
            top_k=state.get("top_k", 5),
        )
        return {
            "depth": 0,
            "evidence": hits,
            "followups": reasoner.generate_followups(
                state["query"],
                hits,
                depth=0,
            ),
        }

    def drift_expand(state: GraphRAGState) -> GraphRAGState:
        depth = state.get("depth", 0) + 1
        evidence = list(state.get("evidence", []))
        for followup in state.get("followups", []):
            evidence.extend(
                retriever.local_search(
                    followup,
                    top_k=state.get("top_k", 5),
                )
            )

        return {
            "depth": depth,
            "evidence": evidence,
            "followups": reasoner.generate_followups(
                state["query"],
                evidence,
                depth=depth,
            ),
        }

    def drift_should_continue(state: GraphRAGState) -> str:
        if state.get("depth", 0) >= state.get("max_depth", 2):
            return "reduce"
        if not state.get("followups"):
            return "reduce"
        return "expand"

    def drift_reduce(state: GraphRAGState) -> GraphRAGState:
        return {
            "answer": reasoner.reduce_drift(
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
