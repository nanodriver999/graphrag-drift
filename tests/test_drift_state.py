from __future__ import annotations

from dataclasses import dataclass, field
from threading import Barrier, get_ident

from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.drift_state import DriftAction, DriftQueryState
from graphrag_drift.llm import DeterministicReasoner
from graphrag_drift.models import (
    CommunityReport,
    DriftActionResult,
    DriftPrimerResult,
    SearchHit,
)


def test_query_state_ranks_high_priority_sibling_before_low_priority_child() -> None:
    state = DriftQueryState()
    root = DriftAction(
        query="root",
        depth=0,
        priority=90,
        answer="primer",
        score=90,
    )
    state.add_action(root)
    state.add_followups("root", ["q1", "q2"], depth=1, priority=90)

    state.complete_action(
        "q1",
        DriftActionResult("q1 answer", 10, ["q1-child"]),
        [SearchHit("e1", "evidence", 1.0)],
    )
    state.add_followups("q1", ["q1-child"], depth=2, priority=10)

    ranked = state.rank_incomplete_actions()

    assert [action.query for action in ranked] == ["q2", "q1-child"]


@dataclass
class RecordingRetriever:
    local_calls: list[str] = field(default_factory=list)
    community_calls: list[str] = field(default_factory=list)

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        self.local_calls.append(query)
        return [SearchHit(f"hit:{query}", f"evidence for {query}", 0.8)]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        self.community_calls.append(query)
        return [CommunityReport(f"c:{query}", f"community context for {query}", 0.9)]


class ScriptedReasoner(DeterministicReasoner):
    action_context_ids: dict[str, list[str]]

    def __init__(self) -> None:
        self.action_context_ids = {}

    def prime_drift(
        self,
        query: str,
        reports: list[CommunityReport],
    ) -> DriftPrimerResult:
        return DriftPrimerResult("primer answer", 90, ["q1", "q2"])

    def answer_drift_action(
        self,
        global_query: str,
        action_query: str,
        hits: list[SearchHit],
        *,
        depth: int,
    ) -> DriftActionResult:
        self.action_context_ids[action_query] = [hit.id for hit in hits]
        if action_query == "q1":
            return DriftActionResult("q1 answer", 10, ["q1-child"])
        if action_query == "q2":
            return DriftActionResult("q2 answer", 80, [])
        return DriftActionResult("child answer", 5, [])


def test_engine_processes_ranked_frontier_with_mixed_local_and_community_context() -> None:
    retriever = RecordingRetriever()
    reasoner = ScriptedReasoner()
    engine = GraphRAGEngine(retriever=retriever, reasoner=reasoner)

    result = engine.drift(
        "root question",
        top_k=1,
        max_depth=2,
        max_actions_per_depth=1,
        community_top_k=1,
    )

    assert retriever.local_calls == ["q1", "q2"]
    assert retriever.community_calls == ["root question", "q1", "q2"]

    assert reasoner.action_context_ids["q1"] == [
        "hit:q1",
        "community:c:q1",
    ]
    assert reasoner.action_context_ids["q2"] == [
        "hit:q2",
        "community:c:q2",
    ]

    assert result["epochs"] == 2
    assert result["completed_action_count"] == 3  # primer + q1 + q2
    assert result["frontier_count"] == 1
    assert result["followups"] == ["q1-child"]
    assert result["actions"]["edges"]
    assert "community:c:q1" in result["actions"]["nodes"][1]["evidence_ids"]


def test_drift_can_disable_community_context_for_followup_actions() -> None:
    retriever = RecordingRetriever()
    reasoner = ScriptedReasoner()
    engine = GraphRAGEngine(retriever=retriever, reasoner=reasoner)

    engine.drift(
        "root question",
        top_k=1,
        max_depth=1,
        max_actions_per_depth=1,
        community_top_k=0,
    )

    assert retriever.community_calls == ["root question"]
    assert reasoner.action_context_ids["q1"] == ["hit:q1"]


class ParallelReasoner(DeterministicReasoner):
    def __init__(self) -> None:
        self.barrier = Barrier(2, timeout=2)
        self.thread_ids: set[int] = set()

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
        self.thread_ids.add(get_ident())
        self.barrier.wait()
        return DriftActionResult(f"{action_query} answer", 80, [])


def test_drift_executes_same_epoch_actions_concurrently() -> None:
    retriever = RecordingRetriever()
    reasoner = ParallelReasoner()
    engine = GraphRAGEngine(retriever=retriever, reasoner=reasoner)

    result = engine.drift(
        "root question",
        top_k=1,
        max_depth=1,
        max_actions_per_depth=2,
        community_top_k=0,
        action_concurrency=2,
    )

    assert result["completed_action_count"] == 3
    assert result["epochs"] == 1
    assert result["action_concurrency"] == 2
    assert len(reasoner.thread_ids) == 2


def test_drift_action_concurrency_one_preserves_sequential_mode() -> None:
    retriever = RecordingRetriever()
    reasoner = ScriptedReasoner()
    engine = GraphRAGEngine(retriever=retriever, reasoner=reasoner)

    result = engine.drift(
        "root question",
        top_k=1,
        max_depth=1,
        max_actions_per_depth=2,
        community_top_k=0,
        action_concurrency=1,
    )

    assert result["action_concurrency"] == 1
    assert result["completed_action_count"] == 3


@dataclass
class PrimerFoldRetriever:
    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        return [SearchHit("local", "fallback", 0.5)]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        reports = [
            CommunityReport("r1", "report 1", 0.9),
            CommunityReport("r2", "report 2", 0.8),
            CommunityReport("r3", "report 3", 0.7),
            CommunityReport("r4", "report 4", 0.6),
        ]
        return reports[:top_k]


class PrimerFoldReasoner(DeterministicReasoner):
    def __init__(self) -> None:
        self.barrier = Barrier(2, timeout=2)
        self.thread_ids: set[int] = set()
        self.fold_ids: list[list[str]] = []

    def prime_drift(
        self,
        query: str,
        reports: list[CommunityReport],
    ) -> DriftPrimerResult:
        ids = [report.id for report in reports]
        self.fold_ids.append(ids)
        self.thread_ids.add(get_ident())
        self.barrier.wait()
        score = 80 if ids[0] == "r1" else 60
        return DriftPrimerResult(
            f"primer for {','.join(ids)}",
            score,
            ["shared", f"followup-{ids[0]}"],
        )


def test_drift_primer_splits_reports_into_parallel_folds_and_aggregates() -> None:
    reasoner = PrimerFoldReasoner()
    engine = GraphRAGEngine(
        retriever=PrimerFoldRetriever(),
        reasoner=reasoner,
    )

    result = engine.drift_primer(
        "root question",
        top_k=4,
        primer_folds=2,
        primer_concurrency=2,
    )

    assert result["primer_fold_count"] == 2
    assert result["primer_score"] == 70
    assert result["primer_report_ids"] == ["r1", "r2", "r3", "r4"]
    assert result["followups"] == ["shared", "followup-r1", "followup-r3"]
    assert "primer for r1,r2" in result["primer_answer"]
    assert "primer for r3,r4" in result["primer_answer"]
    assert len(reasoner.thread_ids) == 2
    assert sorted(reasoner.fold_ids) == [["r1", "r2"], ["r3", "r4"]]


def test_drift_primer_caps_fold_count_to_report_count() -> None:
    engine = GraphRAGEngine(
        retriever=PrimerFoldRetriever(),
        reasoner=DeterministicReasoner(),
    )

    result = engine.drift_primer(
        "root question",
        top_k=2,
        primer_folds=5,
        primer_concurrency=5,
    )

    assert result["primer_fold_count"] == 2
