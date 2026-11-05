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
        rows = [
            SearchHit("entity:1", "entity evidence 1", 1.0),
            SearchHit("chunk:1", "text unit 1", 0.95),
            SearchHit("chunk:2", "text unit 2", 0.90),
            SearchHit("entity:2", "entity evidence 2", 0.85),
            SearchHit("chunk:3", "text unit 3", 0.80),
            SearchHit("entity:3", "entity evidence 3", 0.75),
        ]
        return rows[:top_k]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        self.community_top_ks.append(top_k)
        return [
            CommunityReport(f"community-{index}", f"community report {index}", 0.9)
            for index in range(top_k)
        ]


def test_mixed_local_context_uses_three_way_default_quota() -> None:
    base = FakeRetriever()
    retriever = MixedLocalContextRetriever(
        base,
        community_prop=0.25,
        text_unit_prop=0.50,
    )

    hits = retriever.local_search("q", top_k=4)

    assert base.local_top_ks == [16]
    assert base.community_top_ks == [1]
    assert [hit.id for hit in hits] == [
        "entity:1",
        "chunk:1",
        "chunk:2",
        "community:community-0",
    ]


def test_mixed_local_context_allocation_is_deterministic_for_small_top_k() -> None:
    retriever = MixedLocalContextRetriever(
        FakeRetriever(),
        community_prop=0.25,
        text_unit_prop=0.50,
    )

    assert retriever._allocation(3) == {
        "entity": 1,
        "text_unit": 1,
        "community": 1,
    }


def test_mixed_local_context_keeps_single_result_when_preferred_source_is_missing() -> None:
    @dataclass
    class EntityOnlyRetriever(FakeRetriever):
        def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
            self.local_top_ks.append(top_k)
            return [SearchHit("entity:1", "entity only", 1.0)]

    base = EntityOnlyRetriever()
    retriever = MixedLocalContextRetriever(base)

    hits = retriever.local_search("q", top_k=1)

    assert [hit.id for hit in hits] == ["entity:1"]


def test_mixed_local_context_fills_missing_community_slots_from_local_candidates() -> None:
    @dataclass
    class NoCommunityRetriever(FakeRetriever):
        def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
            self.community_top_ks.append(top_k)
            return []

    base = NoCommunityRetriever()
    retriever = MixedLocalContextRetriever(base)

    hits = retriever.local_search("q", top_k=4)

    assert len(hits) == 4
    assert all(not hit.id.startswith("community:") for hit in hits)


def test_mixed_local_context_can_disable_community_and_text_units() -> None:
    base = FakeRetriever()
    retriever = MixedLocalContextRetriever(
        base,
        community_prop=0.0,
        text_unit_prop=0.0,
    )

    hits = retriever.local_search("q", top_k=3)

    assert [hit.id for hit in hits] == ["entity:1", "entity:2", "entity:3"]
    assert base.community_top_ks == []


@pytest.mark.parametrize(
    ("community_prop", "text_unit_prop"),
    [
        (-0.1, 0.5),
        (1.0, 0.0),
        (0.25, -0.1),
        (0.0, 1.0),
        (0.75, 0.5),
    ],
)
def test_mixed_local_context_rejects_invalid_proportions(
    community_prop: float,
    text_unit_prop: float,
) -> None:
    with pytest.raises(ValueError):
        MixedLocalContextRetriever(
            FakeRetriever(),
            community_prop=community_prop,
            text_unit_prop=text_unit_prop,
        )
