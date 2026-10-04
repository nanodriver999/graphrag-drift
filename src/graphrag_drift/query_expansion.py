from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .global_llm_reasoner import TextGenerator
from .models import CommunityReport, SearchHit
from .retrieval import Retriever


_PARTICLE_SUFFIXES = (
    "으로", "에서", "에게", "부터", "까지", "의", "은", "는", "이", "가",
    "을", "를", "에", "와", "과", "로", "도", "만",
)
_GENERIC_QUERY_TERMS = {
    "주요", "무엇인가", "무엇인지", "설명", "알려줘", "관계", "관련",
    "의무", "책임", "업무", "직무", "역할",
}
_DUTY_TERMS = ("의무", "책임", "업무", "직무", "역할")


def _strip_particle(token: str) -> str:
    for suffix in _PARTICLE_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            return token[: -len(suffix)]
    return token


def _legal_anchor(query: str) -> str | None:
    candidates: list[str] = []
    for token in re.findall(r"[가-힣A-Za-z0-9_]+", query):
        normalized = _strip_particle(token.strip())
        if len(normalized) < 3 or normalized in _GENERIC_QUERY_TERMS:
            continue
        candidates.append(normalized)
    if not candidates:
        return None
    return max(candidates, key=len)


def _deterministic_legal_variants(query: str) -> list[str]:
    if not any(term in query for term in _DUTY_TERMS):
        return []

    anchor = _legal_anchor(query)
    if not anchor:
        return []

    return [f"{anchor} 업무"]


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
            "or close legal terms when helpful. For duty/obligation questions, consider "
            "the statute's wording such as 업무, 직무, 책임, and 준수.\n"
            "Return ONLY a JSON array of strings. Do not answer the question.\n\n"
            f"User question:\n{query}\n"
        )
        raw = self.generator.generate(prompt).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = []

        if not isinstance(payload, list):
            payload = []

        expansions: list[str] = []
        for item in [*_deterministic_legal_variants(query), *payload]:
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
