from __future__ import annotations

from dataclasses import dataclass

from .llm import DeterministicReasoner, Reasoner
from .models import SearchHit
from .retrieval import Retriever


@dataclass
class GraphRAGEngine:
    retriever: Retriever
    reasoner: Reasoner | None = None

    def __post_init__(self) -> None:
        if self.reasoner is None:
            self.reasoner = DeterministicReasoner()

    def local(self, query: str, *, top_k: int = 5) -> dict:
        assert self.reasoner is not None
        hits = self.retriever.local_search(query, top_k=top_k)
        return {
            "evidence": hits,
            "answer": self.reasoner.answer_local(query, hits),
        }

    def global_search(self, query: str, *, top_k: int = 5) -> dict:
        assert self.reasoner is not None
        reports = self.retriever.community_reports(query, top_k=top_k)
        partials = [self.reasoner.map_community(query, report) for report in reports]
        return {
            "partials": partials,
            "answer": self.reasoner.reduce_global(query, partials),
        }

    def drift_primer(self, query: str, *, top_k: int = 5) -> dict:
        assert self.reasoner is not None
        hits = self.retriever.local_search(query, top_k=top_k)
        return {
            "depth": 0,
            "evidence": hits,
            "followups": self.reasoner.generate_followups(query, hits, depth=0),
        }

    def drift_expand(
        self,
        query: str,
        *,
        evidence: list[SearchHit],
        followups: list[str],
        depth: int,
        top_k: int = 5,
    ) -> dict:
        assert self.reasoner is not None
        next_depth = depth + 1
        next_evidence = list(evidence)
        for followup in followups:
            next_evidence.extend(self.retriever.local_search(followup, top_k=top_k))

        return {
            "depth": next_depth,
            "evidence": next_evidence,
            "followups": self.reasoner.generate_followups(
                query,
                next_evidence,
                depth=next_depth,
            ),
        }

    def drift_reduce(self, query: str, evidence: list[SearchHit]) -> str:
        assert self.reasoner is not None
        return self.reasoner.reduce_drift(query, evidence)

    def drift(self, query: str, *, top_k: int = 5, max_depth: int = 2) -> dict:
        state = self.drift_primer(query, top_k=top_k)
        while state["followups"] and state["depth"] < max_depth:
            state = self.drift_expand(
                query,
                evidence=state["evidence"],
                followups=state["followups"],
                depth=state["depth"],
                top_k=top_k,
            )

        state["answer"] = self.drift_reduce(query, state["evidence"])
        return state
