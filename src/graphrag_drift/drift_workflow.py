from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .core import GraphRAGEngine
from .drift_state import DriftAction, DriftQueryState
from .models import CommunityReport, DriftPrimerResult, SearchHit


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
    resume_requested: bool

    primer_answer: str
    primer_score: int
    primer_report_ids: list[str]
    primer_fold_count: int
    primer_source: str
    primer_report_folds: list[list[dict[str, Any]]]
    primer_fold_results: dict[str, dict[str, Any]]
    primer_fold_queue: list[int]
    current_primer_fold: int

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


def _report_to_dict(report: CommunityReport) -> dict[str, Any]:
    return {"id": report.id, "summary": report.summary, "score": report.score}


def _report_from_dict(payload: dict[str, Any]) -> CommunityReport:
    return CommunityReport(
        id=str(payload["id"]),
        summary=str(payload["summary"]),
        score=float(payload.get("score", 1.0)),
    )


def _primer_result_to_dict(result: DriftPrimerResult) -> dict[str, Any]:
    return {
        "intermediate_answer": result.intermediate_answer,
        "score": result.score,
        "follow_up_queries": list(result.follow_up_queries),
    }


def _primer_result_from_dict(payload: dict[str, Any]) -> DriftPrimerResult:
    return DriftPrimerResult(
        intermediate_answer=str(payload.get("intermediate_answer") or ""),
        score=int(payload.get("score", 0)),
        follow_up_queries=[
            str(item)
            for item in payload.get("follow_up_queries", [])
            if str(item).strip()
        ],
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

    Checkpoints are created for each primer fold, frontier initialization, every
    ranked action, epoch finalization, and final reduce. Failed folds/actions retry
    independently; already completed work remains checkpointed.
    """

    def prepare_primer(state: DriftWorkflowState) -> DriftWorkflowState:
        try:
            reports = engine.retriever.community_reports(
                state["query"],
                top_k=state.get("top_k", 5),
            )
            if not reports:
                assert engine.reasoner is not None
                hits = engine.retriever.local_search(
                    state["query"],
                    top_k=state.get("top_k", 5),
                )
                followups = engine.reasoner.generate_followups(
                    state["query"],
                    hits,
                    depth=0,
                )
                return {
                    **_clear_failure("prepare_primer"),
                    "primer_answer": "",
                    "primer_score": 0,
                    "primer_report_ids": [],
                    "primer_fold_count": 0,
                    "primer_source": "local-fallback",
                    "primer_report_folds": [],
                    "primer_fold_results": {},
                    "primer_fold_queue": [],
                    "current_primer_fold": -1,
                    "evidence": [_hit_to_dict(hit) for hit in hits],
                    "followups": followups,
                    "epochs": 0,
                }

            folds = engine._split_primer_reports(
                reports,
                primer_folds=state.get("primer_folds", 5),
            )
        except Exception as exc:
            return _record_failure(state, step="prepare_primer", exc=exc)

        return {
            **_clear_failure("prepare_primer"),
            "primer_report_ids": [report.id for report in reports],
            "primer_fold_count": len(folds),
            "primer_source": "community",
            "primer_report_folds": [
                [_report_to_dict(report) for report in fold]
                for fold in folds
            ],
            "primer_fold_results": {},
            "primer_fold_queue": list(range(len(folds))),
            "current_primer_fold": -1,
            "evidence": [],
            "followups": [],
            "epochs": 0,
        }

    def execute_primer_fold(state: DriftWorkflowState) -> DriftWorkflowState:
        queue = list(state.get("primer_fold_queue", []))
        if not queue:
            return {
                **_clear_failure("execute_primer_fold"),
                "current_primer_fold": -1,
            }

        fold_index = int(queue[0])
        step = f"primer_fold:{fold_index}"
        try:
            assert engine.reasoner is not None
            folds = state.get("primer_report_folds", [])
            fold = [
                _report_from_dict(item)
                for item in folds[fold_index]
            ]
            result = engine.reasoner.prime_drift(
                state["query"],
                fold,
            )
        except Exception as exc:
            failed = _record_failure(state, step=step, exc=exc)
            failed["current_primer_fold"] = fold_index
            return failed

        results = dict(state.get("primer_fold_results", {}))
        results[str(fold_index)] = _primer_result_to_dict(result)
        return {
            **_clear_failure(step),
            "primer_fold_results": results,
            "primer_fold_queue": queue[1:],
            "current_primer_fold": fold_index,
        }

    def aggregate_primer(state: DriftWorkflowState) -> DriftWorkflowState:
        try:
            count = int(state.get("primer_fold_count", 0))
            fold_results = [
                _primer_result_from_dict(
                    state.get("primer_fold_results", {})[str(index)]
                )
                for index in range(count)
            ]
            primer = engine._aggregate_primer_results(fold_results)
        except Exception as exc:
            return _record_failure(state, step="aggregate_primer", exc=exc)

        evidence: list[SearchHit] = []
        if primer.intermediate_answer.strip():
            evidence.append(
                SearchHit(
                    id="drift-primer",
                    text=primer.intermediate_answer.strip(),
                    score=primer.score / 100.0,
                )
            )

        return {
            **_clear_failure("aggregate_primer"),
            "primer_answer": primer.intermediate_answer,
            "primer_score": primer.score,
            "followups": list(primer.follow_up_queries),
            "evidence": [_hit_to_dict(hit) for hit in evidence],
            "current_primer_fold": -1,
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

    def after_prepare_primer(state: DriftWorkflowState) -> str:
        failed_step = state.get("failed_step")
        if failed_step:
            return route_after_retryable_step(
                state,
                success_target="initialize_frontier",
                retry_target="prepare_primer",
            )
        if state.get("primer_source") == "local-fallback":
            return "initialize_frontier"
        if state.get("primer_fold_queue"):
            return "execute_primer_fold"
        return "aggregate_primer"

    def after_primer_fold(state: DriftWorkflowState) -> str:
        failed_step = state.get("failed_step")
        if failed_step:
            attempts = _retry_count(state, failed_step)
            if attempts <= state.get("max_step_retries", 2):
                return "execute_primer_fold"
            return "fail"

        if state.get("primer_fold_queue"):
            return "execute_primer_fold"
        return "aggregate_primer"

    def after_aggregate_primer(state: DriftWorkflowState) -> str:
        return route_after_retryable_step(
            state,
            success_target="initialize_frontier",
            retry_target="aggregate_primer",
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

    def route_from_start(state: DriftWorkflowState) -> str:
        if not state.get("resume_requested"):
            return "prepare_primer"

        if state.get("status") == "completed":
            return "done"

        failed_step = str(state.get("failed_step") or "")
        if failed_step == "prepare_primer":
            return "prepare_primer"
        if failed_step.startswith("primer_fold:"):
            return "execute_primer_fold"
        if failed_step == "aggregate_primer":
            return "aggregate_primer"
        if failed_step.startswith("action:"):
            return "execute_action"
        if failed_step == "reduce":
            return "reduce"

        return "prepare_primer"

    graph = StateGraph(DriftWorkflowState)
    graph.add_node("prepare_primer", prepare_primer)
    graph.add_node("execute_primer_fold", execute_primer_fold)
    graph.add_node("aggregate_primer", aggregate_primer)
    graph.add_node("initialize_frontier", initialize_frontier)
    graph.add_node("plan_epoch", plan_epoch)
    graph.add_node("execute_action", execute_action)
    graph.add_node("finalize_epoch", finalize_epoch)
    graph.add_node("reduce", reduce_answer)
    graph.add_node("fail", fail_run)

    graph.add_conditional_edges(
        START,
        route_from_start,
        {
            "prepare_primer": "prepare_primer",
            "execute_primer_fold": "execute_primer_fold",
            "aggregate_primer": "aggregate_primer",
            "execute_action": "execute_action",
            "reduce": "reduce",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "prepare_primer",
        after_prepare_primer,
        {
            "prepare_primer": "prepare_primer",
            "execute_primer_fold": "execute_primer_fold",
            "aggregate_primer": "aggregate_primer",
            "initialize_frontier": "initialize_frontier",
            "fail": "fail",
        },
    )
    graph.add_conditional_edges(
        "execute_primer_fold",
        after_primer_fold,
        {
            "execute_primer_fold": "execute_primer_fold",
            "aggregate_primer": "aggregate_primer",
            "fail": "fail",
        },
    )
    graph.add_conditional_edges(
        "aggregate_primer",
        after_aggregate_primer,
        {
            "aggregate_primer": "aggregate_primer",
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


def resume_drift_workflow(
    workflow: Any,
    *,
    config: dict[str, Any],
    reset_failed_step_retries: bool = True,
) -> dict[str, Any]:
    """Resume a previously checkpointed failed DRIFT run.

    The latest checkpoint is loaded from the workflow's checkpointer. Completed
    primer folds/actions remain in state; only the failed step is routed again.
    """

    snapshot = workflow.get_state(config)
    values = dict(snapshot.values)
    if not values:
        raise ValueError("No checkpointed DRIFT state exists for this thread")

    if values.get("status") == "completed":
        return materialize_drift_result(values)

    failed_step = str(values.get("failed_step") or "")
    if not failed_step:
        raise ValueError("Checkpointed DRIFT run has no failed step to resume")

    retry_counts = dict(values.get("retry_counts", {}))
    if reset_failed_step_retries:
        retry_counts[failed_step] = 0

    resumed = workflow.invoke(
        {
            "resume_requested": True,
            "status": "running",
            "retry_counts": retry_counts,
        },
        config=config,
    )
    return materialize_drift_result(resumed)


def _initial_drift_state(
    query: str,
    *,
    top_k: int,
    max_depth: int,
    max_actions_per_depth: int,
    community_top_k: int,
    action_concurrency: int,
    primer_folds: int,
    primer_concurrency: int,
    max_step_retries: int,
) -> DriftWorkflowState:
    return {
        "query": query,
        "top_k": top_k,
        "max_depth": max_depth,
        "max_actions_per_depth": max_actions_per_depth,
        "community_top_k": community_top_k,
        "action_concurrency": action_concurrency,
        "primer_folds": primer_folds,
        "primer_concurrency": primer_concurrency,
        "max_step_retries": max_step_retries,
        "resume_requested": False,
        "retry_counts": {},
        "primer_report_folds": [],
        "primer_fold_results": {},
        "primer_fold_queue": [],
        "current_primer_fold": -1,
        "epoch_queue": [],
        "current_action_query": "",
        "status": "running",
        "current_step": "start",
        "last_error": "",
        "failed_step": "",
    }


def run_or_resume_drift_workflow(
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
    checkpointer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Start or continue a persistent DRIFT thread."""

    workflow = build_drift_workflow(engine, checkpointer=checkpointer)
    snapshot = workflow.get_state(config)
    values = dict(snapshot.values)

    if values:
        status = str(values.get("status") or "")
        if status == "completed":
            result = materialize_drift_result(values)
            result["checkpoint_reused"] = True
            return result
        if status == "failed":
            result = resume_drift_workflow(
                workflow,
                config=config,
                reset_failed_step_retries=True,
            )
            result["checkpoint_reused"] = True
            return result
        if snapshot.next:
            resumed = workflow.invoke(None, config=config)
            result = materialize_drift_result(resumed)
            result["checkpoint_reused"] = True
            return result
        raise ValueError(
            "Checkpointed DRIFT thread is running but has no pending node to resume"
        )

    initial = _initial_drift_state(
        query,
        top_k=top_k,
        max_depth=max_depth,
        max_actions_per_depth=max_actions_per_depth,
        community_top_k=community_top_k,
        action_concurrency=action_concurrency,
        primer_folds=primer_folds,
        primer_concurrency=primer_concurrency,
        max_step_retries=max_step_retries,
    )
    state = workflow.invoke(initial, config=config)
    result = materialize_drift_result(state)
    result["checkpoint_reused"] = False
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
    initial = _initial_drift_state(
        query,
        top_k=top_k,
        max_depth=max_depth,
        max_actions_per_depth=max_actions_per_depth,
        community_top_k=community_top_k,
        action_concurrency=action_concurrency,
        primer_folds=primer_folds,
        primer_concurrency=primer_concurrency,
        max_step_retries=max_step_retries,
    )
    state = workflow.invoke(initial, config=config)
    return materialize_drift_result(state)
