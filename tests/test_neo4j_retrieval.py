from __future__ import annotations

from graphrag_drift.neo4j_retrieval import Neo4jRetriever


class FakeRecord:
    def __init__(self, payload):
        self.payload = payload

    def data(self):
        return self.payload


class FakeDriver:
    def execute_query(self, query, parameters_, database_):
        if "MATCH (e:Entity)" in query:
            return (
                [FakeRecord({"id": "e1", "text": "GraphRAG | example", "score": 2.0})],
                None,
                None,
            )
        return (
            [FakeRecord({"id": "c1", "summary": "GraphRAG community", "score": 1.0})],
            None,
            None,
        )

    def close(self):
        pass


def test_neo4j_retriever_maps_rows_to_domain_models():
    retriever = Neo4jRetriever(FakeDriver(), database="neo4j")

    hits = retriever.local_search("GraphRAG", top_k=1)
    reports = retriever.community_reports("GraphRAG", top_k=1)

    assert hits[0].id == "e1"
    assert hits[0].score == 2.0
    assert reports[0].id == "c1"
    assert reports[0].summary == "GraphRAG community"
