from __future__ import annotations

import json
from dataclasses import dataclass, field

from .global_llm_reasoner import TextGenerator
from .models import CommunityReport, SearchHit
from .retrieval import Retriever


@dataclass
class LLMQueryExpandingRetriever:
    """Expand a user query before local retrieval.

    The original query is always retained. The generator only proposes alternate
    search formulations; retrieved evidence still comes from the wrapped retriever.
    """

    retriever: Retriever
    generator: TextGenerator
    max_expansions: int = 3
    last_queries: list[str] = field(default_factory=list, init=False)

    def _expand(self, query: str) -> list[str]:
        prompt = (
            "You are a retrieval query planner for a Korean statute/regulation corpus.\n"
            "Rewrite the user question into at most "
            f"{self.max_expansions} short search queries that can retrieve the governing "
            "clauses. Preserve important named entities. Add likely statutory concepts "
            "or close legal terms when helpful.\n"
            "Return ONLY a JSON array of strings. Do not answer the question.\n\n"
            f"User question:\n{query}\n"
        )
        raw = self.generator.generate(prompt).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(payload, list):
            return []

        expansions: list[str] = []
        for item in payload:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if item and item != query and item not in expansions:
                expansions.append(item)
            if len(expansions) >= self.max_expansions:
                break
        return expansions

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        queries = [query, *self._expand(query)]
        self.last_queries = queries

        candidate_k = max(top_k, 3)
        aggregated: dict[str, tuple[SearchHit, float]] = {}

        for expanded_query in queries:
            hits = self.retriever.local_search(expanded_query, top_k=candidate_k)
            for rank, hit in enumerate(hits):
                vote = 1.0 / (rank + 1)
                previous = aggregated.get(hit.id)
                if previous is None:
                    aggregated[hit.id] = (hit, vote)
                    continue
                previous_hit, previous_vote = previous
                best_hit = hit if hit.score > previous_hit.score else previous_hit
                aggregated[hit.id] = (best_hit, previous_vote + vote)

        ranked = sorted(
            aggregated.values(),
            key=lambda item: (item[0].score + 0.10 * item[1], item[1], item[0].score),
            reverse=True,
        )
        return [hit for hit, _vote in ranked[:top_k]]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        return self.retriever.community_reports(query, top_k=top_k)
