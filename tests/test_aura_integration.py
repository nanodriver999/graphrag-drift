from __future__ import annotations

import os

import pytest

from graphrag_drift.config import Neo4jSettings
from graphrag_drift.neo4j_retrieval import Neo4jRetriever


pytestmark = pytest.mark.skipif(
    not all(os.getenv(name) for name in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")),
    reason="Aura credentials are not configured",
)


def test_aura_connectivity_and_queries():
    settings = Neo4jSettings.from_env()
    retriever = Neo4jRetriever.from_settings(settings)
    try:
        assert isinstance(retriever.local_search("GraphRAG", top_k=1), list)
        assert isinstance(retriever.community_reports("GraphRAG", top_k=1), list)
    finally:
        retriever.close()
