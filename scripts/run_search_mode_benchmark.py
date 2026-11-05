from __future__ import annotations

import argparse
import json
import os

from graphrag_drift import (
    GraphRAGEngine,
    GraphRAGLLMReasoner,
    LLMQueryExpandingRetriever,
    OpenAICompatibleTextGenerator,
)
from graphrag_drift.config import Neo4jSettings
from graphrag_drift.evaluation import GraphRAGModeBenchmark, InstrumentedTextGenerator
from graphrag_drift.neo4j_graphrag_retrieval import Neo4jGraphRAGRetriever


def _generator() -> InstrumentedTextGenerator:
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    model = os.getenv("OPENAI_COMPATIBLE_MODEL")
    if not base_url or not model:
        raise RuntimeError(
            "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL are required"
        )
    return InstrumentedTextGenerator(
        OpenAICompatibleTextGenerator(
            base_url=base_url,
            model=model,
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
            timeout_seconds=float(os.getenv("OPENAI_COMPATIBLE_TIMEOUT_SECONDS", "180")),
            temperature=float(os.getenv("OPENAI_COMPATIBLE_TEMPERATURE", "0")),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Local, Global, and DRIFT GraphRAG modes on the same query."
    )
    parser.add_argument("query")
    parser.add_argument("--local-top-k", type=int, default=4)
    parser.add_argument("--global-top-k", type=int, default=3)
    parser.add_argument("--drift-top-k", type=int, default=2)
    parser.add_argument("--drift-max-depth", type=int, default=1)
    parser.add_argument("--drift-max-actions-per-depth", type=int, default=2)
    parser.add_argument("--drift-community-top-k", type=int, default=1)
    parser.add_argument("--drift-action-concurrency", type=int, default=4)
    parser.add_argument("--drift-primer-folds", type=int, default=5)
    parser.add_argument("--drift-primer-concurrency", type=int, default=5)
    args = parser.parse_args()

    base = Neo4jGraphRAGRetriever.from_settings(Neo4jSettings.from_env())
    try:
        generator = _generator()
        reasoner = GraphRAGLLMReasoner(generator=generator)

        local_retriever = LLMQueryExpandingRetriever(
            retriever=base,
            generator=generator,
        )
        local_engine = GraphRAGEngine(local_retriever, reasoner)
        global_engine = GraphRAGEngine(base, reasoner)
        drift_engine = GraphRAGEngine(base, reasoner)

        result = GraphRAGModeBenchmark(
            local_engine=local_engine,
            global_engine=global_engine,
            drift_engine=drift_engine,
            generator_probe=generator,
        ).run(
            args.query,
            local_top_k=args.local_top_k,
            global_top_k=args.global_top_k,
            drift_top_k=args.drift_top_k,
            drift_max_depth=args.drift_max_depth,
            drift_max_actions_per_depth=args.drift_max_actions_per_depth,
            drift_community_top_k=args.drift_community_top_k,
            drift_action_concurrency=args.drift_action_concurrency,
            drift_primer_folds=args.drift_primer_folds,
            drift_primer_concurrency=args.drift_primer_concurrency,
        )
        result["local_retrieval_queries"] = local_retriever.last_queries
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        base.close()


if __name__ == "__main__":
    main()
