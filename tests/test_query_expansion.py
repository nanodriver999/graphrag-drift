from __future__ import annotations

from dataclasses import dataclass, field

from graphrag_drift.models import CommunityReport, SearchHit
from graphrag_drift.query_expansion import LLMQueryExpandingRetriever


@dataclass
class FakeGenerator:
    response: str
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@dataclass
class FakeRetriever:
    calls: list[str] = field(default_factory=list)

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        self.calls.append(query)
        data = {
            "방사선안전관리자의 주요 의무는 무엇인가?": [
                SearchHit("c1", "방사선안전관리자 자격요건", 0.9),
            ],
            "방사선안전관리자 선임 신고 의무": [
                SearchHit("c2", "방사선안전관리자 선임 및 신고 기한", 0.8),
            ],
            "방사선안전관리 업무 의무": [
                SearchHit("c3", "방사선안전관리 업무 수행 의무", 0.85),
            ],
        }
        return data.get(query, [])[:top_k]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        return []


def test_query_expansion_adds_alternate_retrieval_queries() -> None:
    generator = FakeGenerator(
        '["방사선안전관리자 선임 신고 의무", "방사선안전관리 업무 의무"]'
    )
    wrapped = FakeRetriever()
    retriever = LLMQueryExpandingRetriever(
        retriever=wrapped,
        generator=generator,
        max_expansions=3,
    )

    hits = retriever.local_search(
        "방사선안전관리자의 주요 의무는 무엇인가?",
        top_k=3,
    )

    assert wrapped.calls == [
        "방사선안전관리자의 주요 의무는 무엇인가?",
        "방사선안전관리자 선임 신고 의무",
        "방사선안전관리 업무 의무",
    ]
    assert {hit.id for hit in hits} == {"c1", "c2", "c3"}
    assert retriever.last_queries == wrapped.calls
    assert "Return ONLY a JSON array of strings." in generator.prompts[0]


def test_query_expansion_falls_back_to_original_query_on_invalid_json() -> None:
    generator = FakeGenerator("not json")
    wrapped = FakeRetriever()
    retriever = LLMQueryExpandingRetriever(retriever=wrapped, generator=generator)

    retriever.local_search("방사선안전관리자의 주요 의무는 무엇인가?", top_k=1)

    assert wrapped.calls == ["방사선안전관리자의 주요 의무는 무엇인가?"]
    assert retriever.last_queries == wrapped.calls


def test_query_expansion_deduplicates_duplicate_queries() -> None:
    query = "방사선안전관리자의 주요 의무는 무엇인가?"
    generator = FakeGenerator(
        '["방사선안전관리 업무 의무", "방사선안전관리 업무 의무", '
        '"방사선안전관리자 선임 신고 의무"]'
    )
    wrapped = FakeRetriever()
    retriever = LLMQueryExpandingRetriever(retriever=wrapped, generator=generator)

    retriever.local_search(query, top_k=3)

    assert wrapped.calls == [
        query,
        "방사선안전관리 업무 의무",
        "방사선안전관리자 선임 신고 의무",
    ]
