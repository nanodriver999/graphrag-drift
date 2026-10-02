from __future__ import annotations

from graphrag_drift.config import Neo4jSettings
from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.neo4j_graphrag_retrieval import Neo4jGraphRAGRetriever


def main() -> None:
    settings = Neo4jSettings.from_env()
    retriever = Neo4jGraphRAGRetriever.from_settings(settings, use_hybrid=True)
    try:
        engine = GraphRAGEngine(retriever)
        local = engine.local("iterative local exploration", top_k=3)
        global_result = engine.global_search("GraphRAG search", top_k=3)
        drift = engine.drift("DRIFT exploration", top_k=2, max_depth=2)

        print(f"vector-local: {local['answer']}")
        print(f"vector-global: {global_result['answer']}")
        print(f"vector-drift: {drift['answer']}")

        assert local["evidence"], "Hybrid GraphRAG local search returned no evidence"
        assert global_result["partials"], "Global search returned no community reports"
        assert drift["evidence"], "DRIFT returned no evidence"
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
