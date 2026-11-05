from __future__ import annotations

from dataclasses import dataclass, field

from .models import DriftActionResult, SearchHit


@dataclass
class DriftAction:
    query: str
    depth: int
    parent_query: str | None = None
    priority: float = 0.0
    answer: str | None = None
    score: float | None = None
    follow_ups: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.answer is not None

    def serialize(self) -> dict:
        return {
            "query": self.query,
            "depth": self.depth,
            "parent_query": self.parent_query,
            "priority": self.priority,
            "answer": self.answer,
            "score": self.score,
            "follow_ups": list(self.follow_ups),
            "evidence_ids": list(self.evidence_ids),
            "is_complete": self.is_complete,
        }


@dataclass
class DriftQueryState:
    actions: dict[str, DriftAction] = field(default_factory=dict)
    edges: list[tuple[str, str, float]] = field(default_factory=list)

    def add_action(self, action: DriftAction) -> DriftAction:
        existing = self.actions.get(action.query)
        if existing is not None:
            if action.priority > existing.priority and not existing.is_complete:
                existing.priority = action.priority
            return existing
        self.actions[action.query] = action
        return action

    def add_followups(
        self,
        parent_query: str | None,
        followups: list[str],
        *,
        depth: int,
        priority: float,
    ) -> list[DriftAction]:
        added: list[DriftAction] = []
        for query in followups:
            normalized = query.strip()
            if not normalized:
                continue
            action = self.add_action(
                DriftAction(
                    query=normalized,
                    depth=depth,
                    parent_query=parent_query,
                    priority=priority,
                )
            )
            added.append(action)
            if parent_query is not None:
                edge = (parent_query, action.query, priority)
                if edge not in self.edges:
                    self.edges.append(edge)
        return added

    def complete_action(
        self,
        query: str,
        result: DriftActionResult,
        evidence: list[SearchHit],
    ) -> DriftAction:
        action = self.actions[query]
        action.answer = result.response
        action.score = float(result.score)
        action.follow_ups = list(result.follow_up_queries)
        action.evidence_ids = [hit.id for hit in evidence]
        return action

    def rank_incomplete_actions(self) -> list[DriftAction]:
        incomplete = [action for action in self.actions.values() if not action.is_complete]
        return sorted(
            incomplete,
            key=lambda action: (-action.priority, action.depth, action.query),
        )

    def completed_actions(self) -> list[DriftAction]:
        return [action for action in self.actions.values() if action.is_complete]

    def max_completed_depth(self) -> int:
        completed = self.completed_actions()
        return max((action.depth for action in completed), default=0)

    def serialize(self) -> dict:
        return {
            "nodes": [action.serialize() for action in self.actions.values()],
            "edges": [
                {"source": source, "target": target, "weight": weight}
                for source, target, weight in self.edges
            ],
        }
