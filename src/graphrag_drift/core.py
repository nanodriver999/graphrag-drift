from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .drift_state import DriftAction, DriftQueryState
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
        reports = self.retriever.community_reports(query, top_k=top_k)
        if not reports:
            hits = self.retriever.local_search(query, top_k=top_k)
            return {
                "depth": 0,
                "evidence": hits,
                "followups": self.reasoner.generate_followups(query, hits, depth=0),
                "primer_answer": "",
                "primer_score": 0,
                "primer_report_ids": [],
                "primer_source": "local-fallback",
            }

        primer = self.reasoner.prime_drift(query, reports)
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
            "depth": 0,
            "evidence": evidence,
            "followups": primer.follow_up_queries,
            "primer_answer": primer.intermediate_answer,
            "primer_score": primer.score,
            "primer_report_ids": [report.id for report in reports],
            "primer_source": "community",
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

    def _community_context_hits(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        reports = self.retriever.community_reports(query, top_k=top_k)
        return [
            SearchHit(
                id=f"community:{report.id}",
                text=report.summary,
                score=report.score,
            )
            for report in reports
        ]

    def _drift_mixed_context(
        self,
        query: str,
        *,
        local_top_k: int,
        community_top_k: int,
    ) -> list[SearchHit]:
        hits = self.retriever.local_search(query, top_k=local_top_k)
        hits.extend(
            self._community_context_hits(
                query,
                top_k=community_top_k,
            )
        )
        return hits

    def _execute_drift_action(
        self,
        global_query: str,
        action: DriftAction,
        *,
        top_k: int,
        community_top_k: int,
    ) -> tuple[DriftAction, object, list[SearchHit]]:
        assert self.reasoner is not None
        hits = self._drift_mixed_context(
            action.query,
            local_top_k=top_k,
            community_top_k=community_top_k,
        )
        result = self.reasoner.answer_drift_action(
            global_query,
            action.query,
            hits,
            depth=action.depth,
        )
        return action, result, hits

    def drift_reduce(self, query: str, evidence: list[SearchHit]) -> str:
        assert self.reasoner is not None
        return self.reasoner.reduce_drift(query, evidence)

    def drift(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_depth: int = 2,
        max_actions_per_depth: int = 2,
        community_top_k: int = 1,
        action_concurrency: int = 4,
    ) -> dict:
        assert self.reasoner is not None
        primer_state = self.drift_primer(query, top_k=top_k)
        evidence = list(primer_state["evidence"])
        query_state = DriftQueryState()

        primer_answer = str(primer_state.get("primer_answer") or "")
        primer_score = int(primer_state.get("primer_score", 0))
        primer_source = str(primer_state.get("primer_source") or "")

        if primer_source == "community":
            root = DriftAction(
                query=query,
                depth=0,
                priority=float(primer_score),
                answer=primer_answer,
                score=float(primer_score),
                follow_ups=list(primer_state.get("followups", [])),
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
                list(primer_state.get("followups", [])),
                depth=1,
                priority=fallback_priority,
            )

        epochs = 0
        while epochs < max_depth:
            frontier = query_state.rank_incomplete_actions()
            if not frontier:
                break

            actions = frontier[: max(1, max_actions_per_depth)]
            worker_count = min(max(1, action_concurrency), len(actions))

            if worker_count == 1:
                action_results = [
                    self._execute_drift_action(
                        query,
                        action,
                        top_k=top_k,
                        community_top_k=community_top_k,
                    )
                    for action in actions
                ]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(
                            self._execute_drift_action,
                            query,
                            action,
                            top_k=top_k,
                            community_top_k=community_top_k,
                        )
                        for action in actions
                    ]
                    # Consume futures in ranked action order so state updates remain deterministic.
                    action_results = [future.result() for future in futures]

            for action, action_result, hits in action_results:
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

                if action.depth < max_depth and action_result.follow_up_queries:
                    query_state.add_followups(
                        action.query,
                        action_result.follow_up_queries,
                        depth=action.depth + 1,
                        priority=float(action_result.score),
                    )

            epochs += 1

        remaining = query_state.rank_incomplete_actions()
        result = {
            **primer_state,
            "depth": query_state.max_completed_depth(),
            "evidence": evidence,
            "followups": [action.query for action in remaining],
            "actions": query_state.serialize(),
            "action_count": len(query_state.actions),
            "completed_action_count": len(query_state.completed_actions()),
            "frontier_count": len(remaining),
            "epochs": epochs,
            "action_concurrency": max(1, action_concurrency),
        }
        result["answer"] = self.drift_reduce(query, evidence)
        return result
