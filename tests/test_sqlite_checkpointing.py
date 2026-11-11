from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from graphrag_drift.checkpointing import open_sqlite_checkpointer
from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.drift_workflow import (
    build_drift_workflow,
    resume_drift_workflow,
    run_drift_workflow,
)
from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import CommunityReport, DriftActionResult, DriftPrimerResult, SearchHit


@dataclass
class Retriever:
    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        return [SearchHit(f"hit:{query}", f"evidence:{query}", 0.8)]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        return [CommunityReport("c1", "community", 0.9)]


class FailingActionReasoner(DeterministicReasoner):
    def prime_drift(
        self,
        query: str,
        reports: list[CommunityReport],
    ) -> DriftPrimerResult:
        return DriftPrimerResult("primer", 90, ["q1"])

    def answer_drift_action(
        self,
        global_query: str,
        action_query: str,
        hits: list[SearchHit],
        *,
        depth: int,
    ) -> DriftActionResult:
        raise TimeoutError("provider unavailable")


class ResumeOnlyReasoner(DeterministicReasoner):
    def __init__(self) -> None:
        self.primer_calls = 0
        self.action_calls = 0

    def prime_drift(
        self,
        query: str,
        reports: list[CommunityReport],
    ) -> DriftPrimerResult:
        self.primer_calls += 1
        raise AssertionError("completed primer checkpoint must be reused")

    def answer_drift_action(
        self,
        global_query: str,
        action_query: str,
        hits: list[SearchHit],
        *,
        depth: int,
    ) -> DriftActionResult:
        self.action_calls += 1
        return DriftActionResult("resumed answer", 80, [])


def test_sqlite_checkpoint_resumes_failed_action_across_graph_instances(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "drift.sqlite"
    config = {"configurable": {"thread_id": "persistent-run"}}

    with open_sqlite_checkpointer(checkpoint_path) as saver:
        failed = run_drift_workflow(
            GraphRAGEngine(Retriever(), FailingActionReasoner()),
            "root",
            top_k=1,
            max_depth=1,
            max_actions_per_depth=1,
            community_top_k=0,
            action_concurrency=1,
            primer_folds=1,
            primer_concurrency=1,
            max_step_retries=0,
            checkpointer=saver,
            config=config,
        )

    assert failed["status"] == "failed"
    assert failed["failed_step"] == "action:q1"
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0

    reasoner = ResumeOnlyReasoner()
    with open_sqlite_checkpointer(checkpoint_path) as saver:
        workflow = build_drift_workflow(
            GraphRAGEngine(Retriever(), reasoner),
            checkpointer=saver,
        )
        resumed = resume_drift_workflow(
            workflow,
            config=config,
        )

    assert resumed["status"] == "completed"
    assert resumed["answer"].startswith("DRIFT[root]:")
    assert reasoner.primer_calls == 0
    assert reasoner.action_calls == 1


def test_resume_completed_sqlite_thread_returns_existing_result(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "complete.sqlite"
    config = {"configurable": {"thread_id": "completed-run"}}
    engine = GraphRAGEngine(Retriever(), DeterministicReasoner())

    with open_sqlite_checkpointer(checkpoint_path) as saver:
        completed = run_drift_workflow(
            engine,
            "root",
            top_k=1,
            max_depth=1,
            max_actions_per_depth=1,
            community_top_k=0,
            action_concurrency=1,
            primer_folds=1,
            primer_concurrency=1,
            max_step_retries=0,
            checkpointer=saver,
            config=config,
        )

    with open_sqlite_checkpointer(checkpoint_path) as saver:
        workflow = build_drift_workflow(engine, checkpointer=saver)
        resumed = resume_drift_workflow(
            workflow,
            config=config,
        )

    assert completed["status"] == "completed"
    assert resumed["status"] == "completed"
    assert resumed["answer"] == completed["answer"]
