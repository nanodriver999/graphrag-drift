from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from graphrag_drift import (
    GraphRAGEngine,
    GraphRAGLLMReasoner,
    OpenAICompatibleTextGenerator,
    get_drift_search_profile,
)
from graphrag_drift.benchmark_suite import load_benchmark_cases, run_benchmark_suite
from graphrag_drift.config import Neo4jSettings
from graphrag_drift.drift_profile_comparison import (
    compare_drift_profiles,
    render_drift_profile_comparison_markdown,
)
from graphrag_drift.evaluation import GraphRAGModeBenchmark, InstrumentedTextGenerator
from graphrag_drift.neo4j_graphrag_retrieval import Neo4jGraphRAGRetriever


def _generator() -> OpenAICompatibleTextGenerator:
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
        description="Compare named DRIFT profiles on the same benchmark cases."
    )
    parser.add_argument("--cases", default="benchmarks/legal_smoke.jsonl")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["economy", "microsoft-like"],
    )
    parser.add_argument("--baseline-profile", default="economy")
    parser.add_argument("--drift-top-k", type=int, default=2)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    cases = load_benchmark_cases(args.cases)
    if not cases:
        raise RuntimeError("Benchmark suite contains no cases")

    base_generator = _generator()
    base = Neo4jGraphRAGRetriever.from_settings(Neo4jSettings.from_env())
    try:
        profile_runs: dict[str, dict] = {}

        for profile_name in args.profiles:
            profile = get_drift_search_profile(profile_name)
            probe = InstrumentedTextGenerator(base_generator)
            reasoner = GraphRAGLLMReasoner(
                generator=probe,
                max_followups=profile.max_followups,
            )
            engine = GraphRAGEngine(base, reasoner)
            benchmark = GraphRAGModeBenchmark(
                local_engine=engine,
                global_engine=engine,
                drift_engine=engine,
                generator_probe=probe,
            )

            def runner(query: str, *, _profile=profile, _benchmark=benchmark) -> dict:
                return _benchmark.run_drift_only(
                    query,
                    drift_top_k=args.drift_top_k,
                    drift_profile=_profile.name,
                    drift_max_followups=_profile.max_followups,
                    drift_max_depth=_profile.max_depth,
                    drift_max_actions_per_depth=_profile.max_actions_per_depth,
                    drift_community_top_k=_profile.community_top_k,
                    drift_action_concurrency=_profile.action_concurrency,
                    drift_primer_folds=_profile.primer_folds,
                    drift_primer_concurrency=_profile.primer_concurrency,
                )

            profile_runs[profile.name] = run_benchmark_suite(cases, runner)

        result = compare_drift_profiles(
            profile_runs,
            baseline_profile=args.baseline_profile,
        )
    finally:
        base.close()

    rendered_json = json.dumps(result, ensure_ascii=False, indent=2)
    rendered_md = render_drift_profile_comparison_markdown(result)

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered_json + "\n", encoding="utf-8")
        print(f"Wrote comparison JSON: {path}")
    else:
        print(rendered_json)

    if args.output_markdown:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered_md, encoding="utf-8")
        print(f"Wrote comparison Markdown: {path}")
    elif args.output_json:
        print(rendered_md)


if __name__ == "__main__":
    main()
