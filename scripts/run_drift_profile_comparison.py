from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

from graphrag_drift import (
    GraphRAGEngine,
    GraphRAGLLMReasoner,
    OpenAICompatibleTextGenerator,
    get_drift_search_profile,
    open_sqlite_checkpointer,
    run_or_resume_drift_workflow,
)
from graphrag_drift.benchmark_suite import (
    load_benchmark_cases,
    run_benchmark_suite,
    select_benchmark_cases,
)
from graphrag_drift.config import Neo4jSettings
from graphrag_drift.drift_profile_comparison import (
    compare_drift_profiles,
    render_drift_profile_comparison_markdown,
)
from graphrag_drift.evaluation import GraphRAGModeBenchmark, InstrumentedTextGenerator
from graphrag_drift.neo4j_graphrag_retrieval import Neo4jGraphRAGRetriever
from graphrag_drift.profile_progress import ProfileBenchmarkProgress


def _generator() -> OpenAICompatibleTextGenerator:
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    model = os.getenv("OPENAI_COMPATIBLE_MODEL")
    if not base_url or not model:
        raise RuntimeError(
            "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL are required"
        )
    fallback_models = tuple(
        item.strip()
        for item in os.getenv("OPENAI_COMPATIBLE_MODEL_FALLBACKS", "").split(",")
        if item.strip()
    )
    return OpenAICompatibleTextGenerator(
        base_url=base_url,
        model=model,
        fallback_models=fallback_models,
        api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
        timeout_seconds=float(os.getenv("OPENAI_COMPATIBLE_TIMEOUT_SECONDS", "120")),
        temperature=float(os.getenv("OPENAI_COMPATIBLE_TEMPERATURE", "0")),
        max_retries=int(os.getenv("OPENAI_COMPATIBLE_MAX_RETRIES", "1")),
        retry_backoff_seconds=float(
            os.getenv("OPENAI_COMPATIBLE_RETRY_BACKOFF_SECONDS", "1")
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare named DRIFT profiles on the same benchmark cases."
    )
    parser.add_argument("--cases", default="benchmarks/legal_smoke.jsonl")
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only this benchmark case ID (repeatable); reuse cached others later.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["economy", "microsoft-like"],
    )
    parser.add_argument("--baseline-profile", default="economy")
    parser.add_argument("--drift-top-k", type=int, default=2)
    parser.add_argument(
        "--provider-max-inflight",
        type=int,
        default=1,
        help="Maximum concurrent LLM requests sent to the provider.",
    )
    parser.add_argument("--checkpoint-db", default=None)
    parser.add_argument(
        "--snapshot-sha256",
        default=os.getenv("GRAPHRAG_SNAPSHOT_SHA256"),
        help="SHA-256 of the Neo4j data snapshot; pair with --app-revision.",
    )
    parser.add_argument(
        "--app-revision",
        default=os.getenv("GRAPHRAG_APP_REVISION"),
        help="Immutable application code revision; pair with --snapshot-sha256.",
    )
    parser.add_argument("--progress-json", default=None)
    parser.add_argument("--run-id", default="drift-profile-comparison")
    parser.add_argument("--max-step-retries", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    try:
        cases = select_benchmark_cases(load_benchmark_cases(args.cases), args.case_id)
    except ValueError as exc:
        parser.error(str(exc))
    if not cases:
        raise RuntimeError("Benchmark suite contains no cases")

    base_generator = _generator()
    min_community_term_overlap = int(
        os.getenv("GRAPHRAG_MIN_COMMUNITY_TERM_OVERLAP", "0")
    )
    if min_community_term_overlap < 0:
        parser.error("GRAPHRAG_MIN_COMMUNITY_TERM_OVERLAP must be >= 0")
    raw_allowed = os.getenv("GRAPHRAG_ALLOWED_COMMUNITY_IDS")
    allowed_community_ids = (
        None if raw_allowed is None
        else tuple(sorted({
            item.strip() for item in raw_allowed.split(",") if item.strip()
        }))
    )
    if bool(args.snapshot_sha256) != bool(args.app_revision):
        parser.error("--snapshot-sha256 and --app-revision must be specified together")
    identity = None
    if args.snapshot_sha256:
        identity = {
            "snapshot_sha256": args.snapshot_sha256,
            "app_revision": args.app_revision,
            "as_of_date": os.getenv("GRAPHRAG_AS_OF_DATE", ""),
            "model": base_generator.model,
            "fallback_models": list(base_generator.fallback_models),
            "temperature": base_generator.temperature,
            "provider_max_inflight": args.provider_max_inflight,
            "min_community_term_overlap": min_community_term_overlap,
            "allowed_community_ids": (
                None if allowed_community_ids is None
                else list(allowed_community_ids)
            ),
        }

    # Validate identity before opening Neo4j or resuming any saved LangGraph state.
    progress = (
        ProfileBenchmarkProgress.load(args.progress_json, identity=identity)
        if args.progress_json
        else None
    )
    base = Neo4jGraphRAGRetriever.from_settings(
        Neo4jSettings.from_env(),
        allowed_community_ids=allowed_community_ids,
        min_community_term_overlap=min_community_term_overlap,
    )
    checkpoint_context = (
        open_sqlite_checkpointer(args.checkpoint_db)
        if args.checkpoint_db
        else nullcontext(None)
    )
    try:
        profile_runs: dict[str, dict] = {}

        with checkpoint_context as checkpointer:
            for profile_name in args.profiles:
                profile = get_drift_search_profile(profile_name)
                probe = InstrumentedTextGenerator(
                    base_generator,
                    max_concurrent_requests=args.provider_max_inflight,
                )
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

                def runner(
                    query: str,
                    *,
                    _profile=profile,
                    _benchmark=benchmark,
                    _engine=engine,
                ) -> dict:
                    fingerprint_payload = {
                        "query": query,
                        "profile": _profile.serialize(),
                        "top_k": args.drift_top_k,
                    }
                    if identity is not None:
                        fingerprint_payload["identity"] = identity
                    else:
                        if allowed_community_ids is not None:
                            fingerprint_payload["allowed_community_ids"] = list(
                                allowed_community_ids
                            )
                        fingerprint_payload["min_community_term_overlap"] = (
                            min_community_term_overlap
                        )
                    fingerprint = hashlib.sha256(
                        json.dumps(
                            fingerprint_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest()[:16]
                    thread_id = f"{args.run_id}:{_profile.name}:{fingerprint}"

                    if progress is not None:
                        cached = progress.get(thread_id)
                        if cached is not None:
                            return cached

                    drift_executor = None
                    if checkpointer is not None:
                        config = {"configurable": {"thread_id": thread_id}}

                        def drift_executor() -> dict:
                            result = run_or_resume_drift_workflow(
                                _engine,
                                query,
                                top_k=args.drift_top_k,
                                max_depth=_profile.max_depth,
                                max_actions_per_depth=_profile.max_actions_per_depth,
                                community_top_k=_profile.community_top_k,
                                action_concurrency=_profile.action_concurrency,
                                primer_folds=_profile.primer_folds,
                                primer_concurrency=_profile.primer_concurrency,
                                max_step_retries=args.max_step_retries,
                                checkpointer=checkpointer,
                                config=config,
                            )
                            if result.get("status") == "failed":
                                raise RuntimeError(
                                    "Checkpointed DRIFT step failed: "
                                    f"{result.get('failed_step')}: "
                                    f"{result.get('last_error')}"
                                )
                            return result

                    measured = _benchmark.run_drift_only(
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
                        drift_executor=drift_executor,
                    )
                    if progress is not None:
                        progress.put(thread_id, measured)
                    return measured

                profile_runs[profile.name] = run_benchmark_suite(cases, runner)

        result = compare_drift_profiles(
            profile_runs,
            baseline_profile=args.baseline_profile,
        )
        result["benchmark_identity"] = identity
        result["community_source_filter"] = {
            "min_community_term_overlap": min_community_term_overlap,
            "status": (
                "unrestricted" if allowed_community_ids is None
                else "explicit_allowlist"
            ),
            "allowed_community_ids": (
                None if allowed_community_ids is None
                else list(allowed_community_ids)
            ),
        }
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
