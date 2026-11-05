from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from graphrag_drift.models import CommunityReport, SearchHit
from graphrag_drift.retrieval import MixedLocalContextRetriever


@dataclass
class FakeRetriever:
    local_top_ks: list[int] = field(default_factory=list)
    community_top_ks: list[int] = field(default_factory=list)

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        self.local_top_ks.append(top_k)
        return [
            SearchHit(f"local-{index}", f"local evidence {index}", 1.0 - index * 0.1)
            for index in range(top_k)
        ]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        self.community_top_ks.append(top_k)
        return [
            CommunityReport(f"community-{index}", f"community report {index}", 0.9)
            for index in range(top_k)
        ]


def test_mixed_local_context_reserves_community_quota() -> None:
    base = FakeRetriever()
    retriever = MixedLocalContextRetriever(base, community_prop=0.25)

    hits = retriever.local_search("q", top_k=4)

    assert base.local_top_ks == [3]
    assert base.community_top_ks == [1]
    assert [hit.id for hit in hits] == [
        "local-0",
        "local-1",
        "local-2",
        "community:community-0",
    ]


def test_mixed_local_context_keeps_single_result_local() -> None:
    base = FakeRetriever()
    retriever = MixedLocalContextRetriever(base, community_prop=0.25)

    hits = retriever.local_search("q", top_k=1)

    assert [hit.id for hit in hits] == ["local-0"]
    assert base.community_top_ks == []


def test_mixed_local_context_can_disable_community_reports() -> None:
    base = FakeRetriever()
    retriever = MixedLocalContextRetriever(base, community_prop=0.0)

    hits = retriever.local_search("q", top_k=3)

    assert len(hits) == 3
    assert base.local_top_ks == [3]
    assert base.community_top_ks == []


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.5])
def test_mixed_local_context_rejects_invalid_proportion(value: float) -> None:
    with pytest.raises(ValueError):
        MixedLocalContextRetriever(FakeRetriever(), community_prop=value)
