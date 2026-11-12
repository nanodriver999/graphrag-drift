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
