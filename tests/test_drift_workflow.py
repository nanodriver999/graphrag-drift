from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from langgraph.checkpoint.memory import InMemorySaver

from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.drift_workflow import build_drift_workflow, run_drift_workflow
from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import CommunityReport, DriftActionResult, DriftPrimerResult, SearchHit


@dataclass
class Retriever:
    local_calls: list[str] = field(default_factory=list)

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        self.local_calls.append(query)
        return [SearchHit(f"hit:{query}", f"evidence:{query}", 0.8)]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        return [CommunityReport("c1", "community", 0.9)]


class ScriptedReasoner(DeterministicReasoner):
    def prime_drift(
        self,
        query: str,
        reports: list[CommunityReport],
    ) -> DriftPrimerResult:
        return DriftPrimerResult("primer", 90, ["q1", "q2"])

    def answer_drift_action(
        self,
        global_query: str,
        action_query: str,
        hits: list[SearchHit],
        *,
        depth: int,
    ) -> DriftActionResult:
        if action_query == "q1":
            return DriftActionResult("answer q1", 70, ["q1-child"])
        return DriftActionResult(f"answer {action_query}", 60, [])


class FlakySecondActionReasoner(ScriptedReasoner):
    def __init__(self) -> None:
        self.attempts: Counter[str] = Counter()
        self.q2_failures_remaining = 1

    def answer_drift_action(
        self,
        global_query: str,
        action_query: str,
        hits: list[SearchHit],
        *,
        depth: int,
    ) -> DriftActionResult:
        self.attempts[action_query] += 1
        if action_query == "q2" and self.q2_failures_remaining:
            self.q2_failures_remaining -= 1
            raise TimeoutError("temporary provider timeout")
        return super().answer_drift_action(
            global_query,
            action_query,
            hits,
            depth=depth,
        )


def test_resumable_workflow_matches_ranked_drift_shape() -> None:
    engine = GraphRAGEngine(Retriever(), ScriptedReasoner())

    result = run_drift_workflow(
        engine,
        "root",
        top_k=1,
        max_depth=2,
        max_actions_per_depth=1,
        community_top_k=0,
        action_concurrency=1,
        primer_folds=1,
        primer_concurrency=1,
    )

    assert result["status"] == "completed"
    assert result["epochs"] == 2
    assert result["completed_action_count"] == 3
    assert result["frontier_count"] == 1
    assert result["followups"] == ["q1-child"]
    assert result["actions"]["nodes"]
    assert result["hierarchy"][0]["query"] == "root"
    assert result["answer"].startswith("DRIFT[root]:")


def test_failed_action_retries_only_that_action() -> None:
    reasoner = FlakySecondActionReasoner()
    engine = GraphRAGEngine(Retriever(), reasoner)

    result = run_drift_workflow(
        engine,
        "root",
        top_k=1,
        max_depth=1,
        max_actions_per_depth=2,
        community_top_k=0,
        action_concurrency=1,
        primer_folds=1,
        primer_concurrency=1,
        max_step_retries=1,
    )

    assert result["status"] == "completed"
    assert result["retry_counts"]["action:q2"] == 1
    assert result["failed_step"] == ""
    assert reasoner.attempts["q1"] == 1
    assert reasoner.attempts["q2"] == 2


def test_retry_exhaustion_returns_failed_checkpointed_action_state() -> None:
    class AlwaysFailReasoner(ScriptedReasoner):
        def answer_drift_action(
            self,
            global_query: str,
            action_query: str,
            hits: list[SearchHit],
            *,
            depth: int,
        ) -> DriftActionResult:
            raise TimeoutError("provider remains unavailable")

    result = run_drift_workflow(
        GraphRAGEngine(Retriever(), AlwaysFailReasoner()),
        "root",
        top_k=1,
        max_depth=1,
        max_actions_per_depth=1,
        action_concurrency=1,
        primer_folds=1,
        primer_concurrency=1,
        max_step_retries=1,
    )

    assert result["status"] == "failed"
    assert result["current_step"] == "action:q1"
    assert result["retry_counts"]["action:q1"] == 2
    assert result["current_action_query"] == "q1"
    assert "TimeoutError" in result["last_error"]


def test_langgraph_checkpointer_records_each_action_checkpoint() -> None:
    engine = GraphRAGEngine(Retriever(), ScriptedReasoner())
    checkpointer = InMemorySaver()
    workflow = build_drift_workflow(engine, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "drift-run-1"}}

    result = workflow.invoke(
        {
            "query": "root",
            "top_k": 1,
            "max_depth": 1,
            "max_actions_per_depth": 2,
            "community_top_k": 0,
            "action_concurrency": 1,
            "primer_folds": 1,
            "primer_concurrency": 1,
            "max_step_retries": 1,
            "retry_counts": {},
            "epoch_queue": [],
            "current_action_query": "",
            "status": "running",
            "current_step": "start",
            "last_error": "",
            "failed_step": "",
        },
        config=config,
    )

    history = list(workflow.get_state_history(config))
    action_snapshots = [
        snapshot
        for snapshot in history
        if str(snapshot.values.get("current_step", "")).startswith("action:")
    ]

    assert result["status"] == "completed"
    assert len(action_snapshots) >= 2
    action_steps = {snapshot.values.get("current_step") for snapshot in action_snapshots}
    assert "action:q1" in action_steps
    assert "action:q2" in action_steps
