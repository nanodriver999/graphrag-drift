from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .core import GraphRAGEngine
from .drift_state import DriftAction, DriftQueryState
from .models import SearchHit


DriftRunStatus = Literal["running", "completed", "failed"]


class DriftWorkflowState(TypedDict, total=False):
    query: str
    top_k: int
    max_depth: int
    max_actions_per_depth: int
    community_top_k: int
    action_concurrency: int
    primer_folds: int
    primer_concurrency: int
    max_step_retries: int

    primer_answer: str
    primer_score: int
    primer_report_ids: list[str]
    primer_fold_count: int
    primer_source: str

    evidence: list[dict[str, Any]]
    action_graph: dict[str, Any]
    epochs: int
    epoch_queue: list[str]
    current_action_query: str

    current_step: str
    status: DriftRunStatus
    last_error: str
    failed_step: str
    retry_counts: dict[str, int]

    depth: int
    followups: list[str]
    actions: dict[str, Any]
    hierarchy: list[dict[str, Any]]
    action_count: int
    completed_action_count: int
    frontier_count: int
    answer: str


def _hit_to_dict(hit: SearchHit) -> dict[str, Any]:
    return {"id": hit.id, "text": hit.text, "score": hit.score}


def _hit_from_dict(payload: dict[str, Any]) -> SearchHit:
    return SearchHit(
        id=str(payload["id"]),
        text=str(payload["text"]),
        score=float(payload.get("score", 1.0)),
    )


def _query_state_from_dict(payload: dict[str, Any]) -> DriftQueryState:
    state = DriftQueryState()

    for node in payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        action = DriftAction(
            query=str(node["query"]),
            depth=int(node.get("depth", 0)),
            parent_query=(
                None
                if node.get("parent_query") is None
                else str(node.get("parent_query"))
            ),
            priority=float(node.get("priority", 0.0)),
            answer=(
                None
                if node.get("answer") is None
                else str(node.get("answer"))
            ),
            score=(
                None
                if node.get("score") is None
                else float(node.get("score"))
            ),
            follow_ups=[
                str(item)
                for item in node.get("follow_ups", [])
                if str(item).strip()
            ],
            evidence_ids=[
                str(item)
                for item in node.get("evidence_ids", [])
                if str(item).strip()
            ],
        )
        state.actions[action.query] = action

    for edge in payload.get("edges", []):
        if not isinstance(edge, dict):
            continue
        state.edges.append(
            (
                str(edge["source"]),
                str(edge["target"]),
                float(edge.get("weight", 0.0)),
            )
        )

    return state


def _retry_count(state: DriftWorkflowState, step: str) -> int:
    return int(state.get("retry_counts", {}).get(step, 0))


def _record_failure(
    state: DriftWorkflowState,
    *,
    step: str,
    exc: Exception,
) -> DriftWorkflowState:
    retry_counts = dict(state.get("retry_counts", {}))
    retry_counts[step] = retry_counts.get(step, 0) + 1
    return {
        "current_step": step,
        "status": "running",
        "last_error": f"{type(exc).__name__}: {exc}",
        "failed_step": step,
        "retry_counts": retry_counts,
    }


def _clear_failure(step: str) -> DriftWorkflowState:
    return {
        "current_step": step,
        "status": "running",
        "last_error": "",
        "failed_step": "",
    }


def build_drift_workflow(
    engine: GraphRAGEngine,
    *,
    checkpointer: Any | None = None,
):
    """Build a resumable LangGraph DRIFT workflow.

    Checkpoints are created between primer, frontier initialization, every ranked
    action, epoch finalization, and final reduce. A failed action retries only that
    action; already completed actions remain checkpointed.
    """

    def primer(state: DriftWorkflowState) -> DriftWorkflowState:
        try:
            result = engine.drift_primer(
                state["query"],
                top_k=state.get("top_k", 5),
                primer_folds=state.get("primer_folds", 5),
                primer_concurrency=state.get("primer_concurrency", 5),
            )
        except Exception as exc:
            return _record_failure(state, step="primer", exc=exc)

        return {
            **_clear_failure("primer"),
            "primer_answer": str(result.get("primer_answer") or ""),
            "primer_score": int(result.get("primer_score", 0)),
            "primer_report_ids": list(result.get("primer_report_ids", [])),
            "primer_fold_count": int(result.get("primer_fold_count", 0)),
            "primer_source": str(result.get("primer_source") or ""),
            "evidence": [
                _hit_to_dict(hit)
                for hit in result.get("evidence", [])
            ],
            "followups": list(result.get("followups", [])),
            "epochs": 0,
        }

    def initialize_frontier(state: DriftWorkflowState) -> DriftWorkflowState:
        query_state = DriftQueryState()
        evidence = [
            _hit_from_dict(item)
            for item in state.get("evidence", [])
        ]
        primer_answer = str(state.get("primer_answer") or "")
        primer_score = int(state.get("primer_score", 0))
        primer_source = str(state.get("primer_source") or "")
        followups = list(state.get("followups", []))

        if primer_source == "community":
            root = DriftAction(
                query=state["query"],
                depth=0,
                priority=float(primer_score),
                answer=primer_answer,
                score=float(primer_score),
                follow_ups=followups,
            )
            query_state.add_action(root)
            query_state.add_followups(
                root.query,
                root.follow_ups,
                depth=1,
                priority=float(primer_score),
            )
        else:
            fallback_priority = max(
                (hit.score for hit in evidence),
                default=0.0,
            ) * 100.0
            query_state.add_followups(
                None,
                followups,
                depth=1,
                priority=fallback_priority,
            )

        return {
            **_clear_failure("initialize_frontier"),
            "action_graph": query_state.serialize(),
        }

    def plan_epoch(state: DriftWorkflowState) -> DriftWorkflowState:
        query_state = _query_state_from_dict(
            state.get("action_graph", {"nodes": [], "edges": []})
        )
        frontier = query_state.rank_incomplete_actions()
        queue = [
            action.query
            for action in frontier[: max(1, state.get("max_actions_per_depth", 2))]
        ]
        return {
            **_clear_failure("plan_epoch"),
            "epoch_queue": queue,
            "current_action_query": "",
        }

    def execute_action(state: DriftWorkflowState) -> DriftWorkflowState:
        queue = list(state.get("epoch_queue", []))
        if not queue:
            return {
                **_clear_failure("execute_action"),
                "current_action_query": "",
            }

        action_query = queue[0]
        step = f"action:{action_query}"
        try:
            query_state = _query_state_from_dict(
                state.get("action_graph", {"nodes": [], "edges": []})
            )
            action = query_state.actions[action_query]
            evidence = [
                _hit_from_dict(item)
                for item in state.get("evidence", [])
            ]

            _, action_result, hits = engine._execute_drift_action(
                state["query"],
                action,
                top_k=state.get("top_k", 5),
                community_top_k=state.get("community_top_k", 1),
            )

            completed = query_state.complete_action(
                action.query,
                action_result,
                hits,
            )
            evidence.extend(hits)
            if completed.answer:
                evidence.append(
                    SearchHit(
                        id=f"drift-action:{completed.query}",
                        text=completed.answer,
                        score=(completed.score or 0.0) / 100.0,
                    )
                )

            if (
                action.depth < state.get("max_depth", 2)
                and action_result.follow_up_queries
            ):
                query_state.add_followups(
                    action.query,
                    action_result.follow_up_queries,
                    depth=action.depth + 1,
                    priority=float(action_result.score),
                )

        except Exception as exc:
            failed = _record_failure(state, step=step, exc=exc)
            failed["current_action_query"] = action_query
            return failed

        return {
            **_clear_failure(step),
            "action_graph": query_state.serialize(),
            "evidence": [_hit_to_dict(hit) for hit in evidence],
            "epoch_queue": queue[1:],
            "current_action_query": action_query,
        }

    def finalize_epoch(state: DriftWorkflowState) -> DriftWorkflowState:
        return {
            **_clear_failure("finalize_epoch"),
            "epochs": int(state.get("epochs", 0)) + 1,
            "current_action_query": "",
        }

    def reduce_answer(state: DriftWorkflowState) -> DriftWorkflowState:
        try:
            evidence = [
                _hit_from_dict(item)
                for item in state.get("evidence", [])
            ]
            answer = engine.drift_reduce(state["query"], evidence)
        except Exception as exc:
            return _record_failure(state, step="reduce", exc=exc)

        query_state = _query_state_from_dict(
            state.get("action_graph", {"nodes": [], "edges": []})
        )
        remaining = query_state.rank_incomplete_actions()

        return {
            **_clear_failure("reduce"),
            "status": "completed",
            "depth": query_state.max_completed_depth(),
            "followups": [action.query for action in remaining],
            "actions": query_state.serialize(),
            "hierarchy": query_state.serialize_hierarchy(),
            "action_count": len(query_state.actions),
            "completed_action_count": len(query_state.completed_actions()),
            "frontier_count": len(remaining),
            "answer": answer,
        }

    def fail_run(state: DriftWorkflowState) -> DriftWorkflowState:
        return {
            "status": "failed",
            "current_step": state.get("failed_step", state.get("current_step", "")),
        }

    def route_after_retryable_step(
        state: DriftWorkflowState,
        *,
        success_target: str,
        retry_target: str,
    ) -> str:
        failed_step = state.get("failed_step")
        if not failed_step:
            return success_target

        attempts = _retry_count(state, failed_step)
        if attempts <= state.get("max_step_retries", 2):
            return retry_target
        return "fail"

    def after_primer(state: DriftWorkflowState) -> str:
        return route_after_retryable_step(
            state,
            success_target="initialize_frontier",
            retry_target="primer",
        )

    def after_plan_epoch(state: DriftWorkflowState) -> str:
        if state.get("epoch_queue"):
            return "execute_action"
        return "reduce"

    def after_action(state: DriftWorkflowState) -> str:
        failed_step = state.get("failed_step")
        if failed_step:
            attempts = _retry_count(state, failed_step)
            if attempts <= state.get("max_step_retries", 2):
                return "execute_action"
            return "fail"

        if state.get("epoch_queue"):
            return "execute_action"
        return "finalize_epoch"

    def after_finalize_epoch(state: DriftWorkflowState) -> str:
        query_state = _query_state_from_dict(
            state.get("action_graph", {"nodes": [], "edges": []})
        )
        if (
            int(state.get("epochs", 0)) >= state.get("max_depth", 2)
            or not query_state.rank_incomplete_actions()
        ):
            return "reduce"
        return "plan_epoch"

    def after_reduce(state: DriftWorkflowState) -> str:
        return route_after_retryable_step(
            state,
            success_target="done",
            retry_target="reduce",
        )

    graph = StateGraph(DriftWorkflowState)
    graph.add_node("primer", primer)
    graph.add_node("initialize_frontier", initialize_frontier)
    graph.add_node("plan_epoch", plan_epoch)
    graph.add_node("execute_action", execute_action)
    graph.add_node("finalize_epoch", finalize_epoch)
    graph.add_node("reduce", reduce_answer)
    graph.add_node("fail", fail_run)

    graph.add_edge(START, "primer")
    graph.add_conditional_edges(
        "primer",
        after_primer,
        {
            "primer": "primer",
            "initialize_frontier": "initialize_frontier",
            "fail": "fail",
        },
    )
    graph.add_edge("initialize_frontier", "plan_epoch")
    graph.add_conditional_edges(
        "plan_epoch",
        after_plan_epoch,
        {
            "execute_action": "execute_action",
            "reduce": "reduce",
        },
    )
    graph.add_conditional_edges(
        "execute_action",
        after_action,
        {
            "execute_action": "execute_action",
            "finalize_epoch": "finalize_epoch",
            "fail": "fail",
        },
    )
    graph.add_conditional_edges(
        "finalize_epoch",
        after_finalize_epoch,
        {
            "plan_epoch": "plan_epoch",
            "reduce": "reduce",
        },
    )
    graph.add_conditional_edges(
        "reduce",
        after_reduce,
        {
            "reduce": "reduce",
            "done": END,
            "fail": "fail",
        },
    )
    graph.add_edge("fail", END)

    return graph.compile(checkpointer=checkpointer)


def materialize_drift_result(state: DriftWorkflowState) -> dict[str, Any]:
    result: dict[str, Any] = dict(state)
    result["evidence"] = [
        _hit_from_dict(item)
        if isinstance(item, dict)
        else item
        for item in state.get("evidence", [])
    ]
    result.pop("action_graph", None)
    return result


def run_drift_workflow(
    engine: GraphRAGEngine,
    query: str,
    *,
    top_k: int = 5,
    max_depth: int = 2,
    max_actions_per_depth: int = 2,
    community_top_k: int = 1,
    action_concurrency: int = 4,
    primer_folds: int = 5,
    primer_concurrency: int = 5,
    max_step_retries: int = 2,
    checkpointer: Any | None = None,
    config: dict[str, Any] | None = None,
) -> DriftWorkflowState:
    workflow = build_drift_workflow(
        engine,
        checkpointer=checkpointer,
    )
    initial: DriftWorkflowState = {
        "query": query,
        "top_k": top_k,
        "max_depth": max_depth,
        "max_actions_per_depth": max_actions_per_depth,
        "community_top_k": community_top_k,
        "action_concurrency": action_concurrency,
        "primer_folds": primer_folds,
        "primer_concurrency": primer_concurrency,
        "max_step_retries": max_step_retries,
        "retry_counts": {},
        "epoch_queue": [],
        "current_action_query": "",
        "status": "running",
        "current_step": "start",
        "last_error": "",
        "failed_step": "",
    }
    state = workflow.invoke(initial, config=config)
    return materialize_drift_result(state)
