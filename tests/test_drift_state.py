from __future__ import annotations

from dataclasses import dataclass, field

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
    calls: list[str] = field(default_factory=list)

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        self.calls.append(query)
        return [SearchHit(f"hit:{query}", f"evidence for {query}", 0.8)]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        return [CommunityReport("c1", "broad community context", 0.9)]


class ScriptedReasoner(DeterministicReasoner):
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
        if action_query == "q1":
            return DriftActionResult("q1 answer", 10, ["q1-child"])
        if action_query == "q2":
            return DriftActionResult("q2 answer", 80, [])
        return DriftActionResult("child answer", 5, [])


def test_engine_processes_ranked_frontier_across_epochs() -> None:
    retriever = RecordingRetriever()
    engine = GraphRAGEngine(retriever=retriever, reasoner=ScriptedReasoner())

    result = engine.drift(
        "root question",
        top_k=1,
        max_depth=2,
        max_actions_per_depth=1,
    )

    assert retriever.calls == ["q1", "q2"]
    assert result["epochs"] == 2
    assert result["completed_action_count"] == 3  # primer + q1 + q2
    assert result["frontier_count"] == 1
    assert result["followups"] == ["q1-child"]
    assert result["actions"]["edges"]
