from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from graphrag_drift import (
    DRIFT_SEARCH_PROFILES,
    GraphRAGEngine,
    GraphRAGLLMReasoner,
    LLMQueryExpandingRetriever,
    MixedLocalContextRetriever,
    OpenAICompatibleTextGenerator,
    resolve_drift_search_profile,
)
from graphrag_drift.benchmark_suite import load_benchmark_cases, run_benchmark_suite
from graphrag_drift.config import Neo4jSettings
from graphrag_drift.evaluation import GraphRAGModeBenchmark, InstrumentedTextGenerator
from graphrag_drift.neo4j_graphrag_retrieval import Neo4jGraphRAGRetriever


def _probe() -> InstrumentedTextGenerator:
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
        description="Run a JSONL benchmark suite across Local, Global, and DRIFT."
    )
    parser.add_argument(
        "--cases",
        default="benchmarks/legal_smoke.jsonl",
        help="JSONL benchmark cases.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--local-top-k", type=int, default=4)
    parser.add_argument("--local-community-prop", type=float, default=0.25)
    parser.add_argument("--local-text-unit-prop", type=float, default=0.50)
    parser.add_argument("--global-top-k", type=int, default=3)
    parser.add_argument("--global-map-concurrency", type=int, default=4)
    parser.add_argument("--drift-top-k", type=int, default=2)
    parser.add_argument(
        "--drift-profile",
        choices=sorted(DRIFT_SEARCH_PROFILES),
        default="economy",
    )
    parser.add_argument("--drift-max-depth", type=int, default=None)
    parser.add_argument("--drift-max-actions-per-depth", type=int, default=None)
    parser.add_argument("--drift-community-top-k", type=int, default=None)
    parser.add_argument("--drift-action-concurrency", type=int, default=None)
    parser.add_argument("--drift-primer-folds", type=int, default=None)
    parser.add_argument("--drift-primer-concurrency", type=int, default=None)
    parser.add_argument("--drift-max-followups", type=int, default=None)
    args = parser.parse_args()

    cases = load_benchmark_cases(args.cases)
    if not cases:
        raise RuntimeError("Benchmark suite contains no cases")

    drift_profile = resolve_drift_search_profile(
        args.drift_profile,
        max_depth=args.drift_max_depth,
        max_actions_per_depth=args.drift_max_actions_per_depth,
        community_top_k=args.drift_community_top_k,
        action_concurrency=args.drift_action_concurrency,
        primer_folds=args.drift_primer_folds,
        primer_concurrency=args.drift_primer_concurrency,
        max_followups=args.drift_max_followups,
    )

    base = Neo4jGraphRAGRetriever.from_settings(Neo4jSettings.from_env())
    try:
        generator = _probe()
        reasoner = GraphRAGLLMReasoner(
            generator=generator,
            max_followups=drift_profile.max_followups,
        )
        mixed_local = MixedLocalContextRetriever(
            retriever=base,
            community_prop=args.local_community_prop,
            text_unit_prop=args.local_text_unit_prop,
        )
        local_retriever = LLMQueryExpandingRetriever(
            retriever=mixed_local,
            generator=generator,
        )
        benchmark = GraphRAGModeBenchmark(
            local_engine=GraphRAGEngine(local_retriever, reasoner),
            global_engine=GraphRAGEngine(base, reasoner),
            drift_engine=GraphRAGEngine(base, reasoner),
            generator_probe=generator,
        )

        def runner(query: str) -> dict:
            return benchmark.run(
                query,
                local_top_k=args.local_top_k,
                local_community_prop=args.local_community_prop,
                local_text_unit_prop=args.local_text_unit_prop,
                global_top_k=args.global_top_k,
                global_map_concurrency=args.global_map_concurrency,
                drift_top_k=args.drift_top_k,
                drift_profile=drift_profile.name,
                drift_max_followups=drift_profile.max_followups,
                drift_max_depth=drift_profile.max_depth,
                drift_max_actions_per_depth=drift_profile.max_actions_per_depth,
                drift_community_top_k=drift_profile.community_top_k,
                drift_action_concurrency=drift_profile.action_concurrency,
                drift_primer_folds=drift_profile.primer_folds,
                drift_primer_concurrency=drift_profile.primer_concurrency,
            )

        result = run_benchmark_suite(cases, runner)
    finally:
        base.close()

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote benchmark result: {path}")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
