from __future__ import annotations

import pytest

from graphrag_drift.neo4j_graphrag_retrieval import (
    CHUNK_LEXICAL_QUERY,
    ENTITY_HYBRID_QUERY,
    ENTITY_VECTOR_QUERY,
    Neo4jGraphRAGRetriever,
)


class CapturingDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []

    def execute_query(self, query: str, *, parameters_: dict, database_: str | None):
        self.calls.append((query, parameters_, database_))
        return [
            {"id": "entity:law-role", "text": "role", "score": 0.8}
        ], None, None


@pytest.mark.parametrize("hybrid", [True, False])
def test_entity_expansion_filters_attached_chunks_by_effective_date(hybrid: bool) -> None:
    driver = CapturingDriver()
    retriever = Neo4jGraphRAGRetriever(
        driver=driver, database="neo4j", use_hybrid=hybrid,
        as_of_date="2026-10-05",
    )
    hits = retriever._search_entities(
        "방사선안전관리자", vector=[0.1, 0.2], candidate_k=3,
    )
    assert hits[0].id == "entity:law-role"
    query, params, db = driver.calls[0]
    assert db == "neo4j"
    assert params["as_of_date"] == "20261005"
    assert "OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(node)" in query
    assert (
        "WHERE coalesce(chunk.officialEffectiveDate, '') = ''\n"
        "   OR chunk.officialEffectiveDate <= $as_of_date"
    ) in query
    if hybrid:
        assert query == ENTITY_HYBRID_QUERY
    else:
        assert query == ENTITY_VECTOR_QUERY


def test_lexical_article_sibling_expansion_respects_effective_date() -> None:
    assert "sibling.articleKey = article_key" in CHUNK_LEXICAL_QUERY
    assert "sibling.officialEffectiveDate <= $as_of_date" in CHUNK_LEXICAL_QUERY


def test_entity_query_preserves_undated_legacy_chunks() -> None:
    # The conservative behavior retains undated material, but explicitly
    # excludes future-dated chunks. Undated entities are not certified legal
    # sources by this query and must be audited separately.
    for query in (ENTITY_HYBRID_QUERY, ENTITY_VECTOR_QUERY):
        assert "coalesce(chunk.officialEffectiveDate, '') = ''" in query
