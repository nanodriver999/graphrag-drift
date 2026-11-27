"""Integration test against a disposable localhost Neo4j, never Aura."""
from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from graphrag_drift.official_law_corpus import (
    OfficialLawVersionRequest, write_official_law_corpus,
)
from graphrag_drift.official_law_corpus_ingestion import (
    import_verified_official_corpus, prepare_official_corpus_chunks,
)
from graphrag_drift.verified_official_ingestion import validate_local_derived_target

from test_official_law_corpus_ingestion import XML


@pytest.mark.skipif(
    not os.environ.get("OFFICIAL_LAW_LOCAL_INTEGRATION"),
    reason="Requires explicitly provisioned disposable localhost Neo4j",
)
def test_atomic_official_law_batch_and_search_on_local_neo4j(tmp_path):
    uri = os.environ["LOCAL_NEO4J_URI"]
    validate_local_derived_target(uri)
    root = tmp_path / "official"
    write_official_law_corpus(
        root, XML,
        OfficialLawVersionRequest("011435", "286119", "2026-05-19", "2026-10-05"),
        required_articles=("53-3",),
    )
    _, chunks = prepare_official_corpus_chunks(root, required_articles=("53-3",))
    with GraphDatabase.driver(
        uri, auth=("neo4j", os.environ["LOCAL_NEO4J_PASSWORD"])
    ) as driver:
        driver.verify_connectivity()
        with driver.session(database="neo4j") as session:
            session.run("MATCH (n:Chunk) DETACH DELETE n").consume()
            session.run(
                "CREATE (:Chunk {articleKey: $key, sourceChunkId: 'bad', "
                "text: 'wrong', sourceResponseSha256: 'wrong'})",
                key=chunks[1]["articleKey"],
            ).consume()

        # On conflict with the SECOND article, the first must be rolled back.
        with pytest.raises(ValueError, match="Manual source reconciliation"):
            import_verified_official_corpus(driver, chunks, database="neo4j")
        with driver.session(database="neo4j") as session:
            count = session.run(
                "MATCH (c:Chunk) WHERE c.articleKey = $key RETURN count(c) AS n",
                key=chunks[0]["articleKey"],
            ).single()["n"]
            assert count == 0
            session.run("MATCH (n:Chunk) DETACH DELETE n").consume()

        result = import_verified_official_corpus(driver, chunks, database="neo4j")
        assert result["created"] == 2
        retry = import_verified_official_corpus(driver, chunks, database="neo4j")
        assert retry["already_present"] == 2
        with driver.session(database="neo4j") as session:
            session.run(
                "CREATE FULLTEXT INDEX official_law_test_fulltext IF NOT EXISTS "
                "FOR (n:Chunk) ON EACH [n.text, n.content]"
            ).consume()
            session.run("CALL db.awaitIndexes(120)").consume()
            rows = list(session.run(
                "CALL db.index.fulltext.queryNodes("
                "'official_law_test_fulltext', '방사선안전관리자') "
                "YIELD node RETURN node.articleKey AS article_key"
            ))
            assert any(row["article_key"].endswith(":art:53-3") for row in rows)
            future = session.run(
                "MATCH (n:Chunk) WHERE n.articleKey ENDS WITH ':art:99' "
                "RETURN count(n) AS n"
            ).single()["n"]
            assert future == 0
