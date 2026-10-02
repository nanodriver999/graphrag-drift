from __future__ import annotations

from graphrag_drift.config import Neo4jSettings
from graphrag_drift.core import GraphRAGEngine
from graphrag_drift.neo4j_retrieval import Neo4jRetriever


def main() -> None:
    settings = Neo4jSettings.from_env()
    retriever = Neo4jRetriever.from_settings(settings)
    try:
        engine = GraphRAGEngine(retriever)
        for mode, result in (
            ("local", engine.local("GraphRAG", top_k=3)),
            ("global", engine.global_search("GraphRAG", top_k=3)),
            ("drift", engine.drift("DRIFT", top_k=2, max_depth=2)),
        ):
            print(f"{mode}: {result['answer']}")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
