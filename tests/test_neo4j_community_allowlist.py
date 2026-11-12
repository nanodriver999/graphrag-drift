from __future__ import annotations

from graphrag_drift.neo4j_graphrag_retrieval import Neo4jGraphRAGRetriever
from graphrag_drift.neo4j_retrieval import COMMUNITY_REPORT_QUERY


class FakeDriver:
    def __init__(self) -> None:
        self.params: list[dict] = []

    def execute_query(self, query: str, *, parameters_: dict, database_: str | None):
        assert query == COMMUNITY_REPORT_QUERY
        self.params.append(parameters_)
        return [
            {"id": "leiden-1", "summary": "report summary", "score": 0.8},
        ], None, None


def test_date_audited_allowlist_is_bound_to_neo4j_report_query() -> None:
    fake = FakeDriver()
    retriever = Neo4jGraphRAGRetriever(
        driver=fake,
        database="neo4j",
        allowed_community_ids=("leiden-1", "community-search"),
    )
    reports = retriever.community_reports("safety", top_k=3)
    assert reports[0].id == "leiden-1"
    assert fake.params[0]["allowed_community_ids"] == [
        "leiden-1", "community-search",
    ]
    assert "$allowed_community_ids IS NULL" in COMMUNITY_REPORT_QUERY
    assert "c.id IN $allowed_community_ids" in COMMUNITY_REPORT_QUERY


def test_empty_allowlist_means_no_community_reports_not_unrestricted() -> None:
    fake = FakeDriver()
    retriever = Neo4jGraphRAGRetriever(
        driver=fake, allowed_community_ids=(),
    )
    retriever.community_reports("safety")
    assert fake.params[0]["allowed_community_ids"] == []


def test_missing_allowlist_preserves_legacy_community_report_search() -> None:
    fake = FakeDriver()
    retriever = Neo4jGraphRAGRetriever(driver=fake)
    retriever.community_reports("safety")
    assert fake.params[0]["allowed_community_ids"] is None



def test_community_primer_relevance_filter_binds_normalized_query_terms() -> None:
    fake = FakeDriver()
    retriever = Neo4jGraphRAGRetriever(
        driver=fake, min_community_term_overlap=1,
        allowed_community_ids=("community-search", "leiden-1"),
    )
    retriever.community_reports("방사선안전관리자의 주요 의무", top_k=3)
    assert fake.params[0]["min_term_overlap"] == 1
    assert "방사선안전관리자" in fake.params[0]["search_terms"]
    assert "방사선안전관리자의" in fake.params[0]["search_terms"]
    assert "$min_term_overlap = 0 OR matched_terms >= $min_term_overlap" in (
        COMMUNITY_REPORT_QUERY
    )
    assert "coalesce(report.summary, c.summary, '')" in COMMUNITY_REPORT_QUERY


def test_default_community_query_does_not_require_lexical_overlap() -> None:
    fake = FakeDriver()
    retriever = Neo4jGraphRAGRetriever(driver=fake)
    retriever.community_reports("radiation safety")
    assert fake.params[0]["min_term_overlap"] == 0


def test_negative_community_overlap_threshold_is_rejected() -> None:
    import pytest

    fake = FakeDriver()
    retriever = Neo4jGraphRAGRetriever(
        driver=fake, min_community_term_overlap=-1
    )
    with pytest.raises(ValueError, match="must be >= 0"):
        retriever.community_reports("radiation safety")
    assert not fake.params
