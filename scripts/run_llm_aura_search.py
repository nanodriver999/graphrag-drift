from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from graphrag_drift import (
    GraphRAGEngine,
    GraphRAGLLMReasoner,
    LLMQueryExpandingRetriever,
    OpenAICompatibleTextGenerator,
)
from graphrag_drift.config import Neo4jSettings
from graphrag_drift.neo4j_graphrag_retrieval import Neo4jGraphRAGRetriever


def _generator_from_env() -> OpenAICompatibleTextGenerator:
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    model = os.getenv("OPENAI_COMPATIBLE_MODEL")
    if not base_url or not model:
        raise RuntimeError(
            "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL are required"
        )
    return OpenAICompatibleTextGenerator(
        base_url=base_url,
        model=model,
        api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
        timeout_seconds=float(os.getenv("OPENAI_COMPATIBLE_TIMEOUT_SECONDS", "180")),
        temperature=float(os.getenv("OPENAI_COMPATIBLE_TEMPERATURE", "0")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-backed Local or DRIFT GraphRAG search against Neo4j Aura."
    )
    parser.add_argument("mode", choices=["local", "drift"])
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--no-query-expansion", action="store_true")
    args = parser.parse_args()

    base_retriever = Neo4jGraphRAGRetriever.from_settings(Neo4jSettings.from_env())
    try:
        generator = _generator_from_env()
        reasoner = GraphRAGLLMReasoner(generator=generator)

        query_expander = None
        retriever = base_retriever
        if args.mode == "local" and not args.no_query_expansion:
            query_expander = LLMQueryExpandingRetriever(
                retriever=base_retriever,
                generator=generator,
            )
            retriever = query_expander

        engine = GraphRAGEngine(retriever=retriever, reasoner=reasoner)
        if args.mode == "local":
            result = engine.local(args.query, top_k=args.top_k)
        else:
            result = engine.drift(
                args.query,
                top_k=args.top_k,
                max_depth=args.max_depth,
            )

        serializable = dict(result)
        if "evidence" in serializable:
            serializable["evidence"] = [asdict(hit) for hit in serializable["evidence"]]
        if query_expander is not None:
            serializable["retrieval_queries"] = query_expander.last_queries
        print(json.dumps(serializable, ensure_ascii=False, indent=2))
    finally:
        base_retriever.close()


if __name__ == "__main__":
    main()
