from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock, Semaphore
from time import perf_counter
from typing import Any

from .core import GraphRAGEngine
from .global_llm_reasoner import TextGenerator
from .source_grounding import collect_article_sources


@dataclass
class InstrumentedTextGenerator:
    generator: TextGenerator
    call_count: int = 0
    total_seconds: float = 0.0
    prompts: list[str] = field(default_factory=list)
    max_concurrent_requests: int | None = None
    _lock: Any = field(default_factory=Lock, init=False, repr=False)
    _semaphore: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_concurrent_requests is not None:
            if self.max_concurrent_requests < 1:
                raise ValueError("max_concurrent_requests must be >= 1")
            self._semaphore = Semaphore(self.max_concurrent_requests)

    def reset(self) -> None:
        with self._lock:
            self.call_count = 0
            self.total_seconds = 0.0
            self.prompts.clear()
        reset = getattr(self.generator, "reset_model_usage", None)
        if callable(reset):
            reset()

    def successful_model_usage(self) -> dict[str, int] | None:
        """Return actual successful provider model IDs when supported."""
        getter = getattr(self.generator, "successful_model_usage", None)
        return getter() if callable(getter) else None

    def generate(self, prompt: str) -> str:
        if self._semaphore is None:
            return self._generate_instrumented(prompt)

        with self._semaphore:
            return self._generate_instrumented(prompt)

    def _generate_instrumented(self, prompt: str) -> str:
        start = perf_counter()
        try:
            return self.generator.generate(prompt)
        finally:
            elapsed = perf_counter() - start
            with self._lock:
                self.call_count += 1
                self.total_seconds += elapsed
                self.prompts.append(prompt)


@dataclass
class GraphRAGModeBenchmark:
    local_engine: GraphRAGEngine
    global_engine: GraphRAGEngine
    drift_engine: GraphRAGEngine
    generator_probe: InstrumentedTextGenerator

    def _metrics(
        self,
        mode: str,
        result: dict[str, Any],
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        evidence_count = 0
        unique_evidence_count = 0
        if mode == "local":
            evidence = result.get("evidence", [])
            evidence_count = len(evidence)
            unique_evidence_count = len({hit.id for hit in evidence})
        elif mode == "global":
            partials = result.get("partials", [])
            evidence_count = len(partials)
            unique_evidence_count = len(partials)
        elif mode == "drift":
            evidence = result.get("evidence", [])
            evidence_count = len(evidence)
            unique_evidence_count = len({hit.id for hit in evidence})

        metrics: dict[str, Any] = {
            "mode": mode,
            "wall_ms": round(elapsed_seconds * 1000, 3),
            "llm_calls": self.generator_probe.call_count,
            "llm_time_ms": round(self.generator_probe.total_seconds * 1000, 3),
            "evidence_count": evidence_count,
            "unique_evidence_count": unique_evidence_count,
            "answer": str(result.get("answer") or ""),
        }
        usage = self.generator_probe.successful_model_usage()
        if usage is not None:
            metrics["successful_model_calls"] = usage
            primary = getattr(self.generator_probe.generator, "model", "")
            metrics["used_fallback_model"] = any(
                model != primary for model in usage
            )
        if mode == "global":
            metrics["map_concurrency"] = int(result.get("map_concurrency", 1))
            metrics["reduce_char_budget"] = int(result.get("reduce_char_budget", 0))
            metrics["reduce_partial_count"] = int(result.get("reduce_partial_count", 0))
            metrics["reduce_char_count"] = int(result.get("reduce_char_count", 0))
        if mode == "drift":
            # A DRIFT run contains fetched evidence plus LLM-synthesized primer
            # and action responses. Do not report the latter as retrieved facts.
            fetched = [
                hit for hit in result.get("evidence", [])
                if hit.id != "drift-primer"
                and not hit.id.startswith("drift-action:")
            ]
            metrics["retrieved_evidence_count"] = len(fetched)
            metrics["unique_retrieved_evidence_count"] = len(
                {hit.id for hit in fetched}
            )
            metrics["synthesized_context_count"] = evidence_count - len(fetched)
            # Preserve only explicit versioned law-article source IDs, never
            # LLM-synthesized primers, action answers or anonymous entity hits.
            sources = collect_article_sources(hit.id for hit in fetched)
            metrics["direct_article_sources"] = sources
            metrics["unique_direct_article_source_count"] = len(sources)
            metrics["depth"] = int(result.get("depth", 0))
            metrics["remaining_followups"] = len(result.get("followups", []))
            metrics["primer_source"] = str(result.get("primer_source") or "")
            metrics["primer_score"] = int(result.get("primer_score", 0))
            metrics["primer_report_count"] = len(result.get("primer_report_ids", []))
            metrics["primer_fold_count"] = int(result.get("primer_fold_count", 0))
            metrics["action_concurrency"] = int(result.get("action_concurrency", 1))
            metrics["primer_concurrency"] = int(result.get("primer_concurrency", 1))
            metrics["action_count"] = int(result.get("action_count", 0))
            metrics["completed_action_count"] = int(result.get("completed_action_count", 0))
            metrics["frontier_count"] = int(result.get("frontier_count", 0))
            metrics["epochs"] = int(result.get("epochs", 0))
            metrics["checkpoint_reused"] = bool(result.get("checkpoint_reused", False))
            metrics["run_status"] = str(result.get("status") or "completed")
        return metrics

    def _run(self, mode: str, fn) -> dict[str, Any]:
        self.generator_probe.reset()
        start = perf_counter()
        result = fn()
        elapsed = perf_counter() - start
        return self._metrics(mode, result, elapsed)

    def run_drift_only(
        self,
        query: str,
        *,
        drift_top_k: int = 5,
        drift_profile: str = "economy",
        drift_max_followups: int = 2,
        drift_max_depth: int = 2,
        drift_max_actions_per_depth: int = 2,
        drift_community_top_k: int = 1,
        drift_action_concurrency: int = 4,
        drift_primer_folds: int = 5,
        drift_primer_concurrency: int = 5,
        drift_executor: Any | None = None,
    ) -> dict[str, Any]:
        if drift_executor is None:
            drift_executor = lambda: self.drift_engine.drift(
                query,
                top_k=drift_top_k,
                max_depth=drift_max_depth,
                max_actions_per_depth=drift_max_actions_per_depth,
                community_top_k=drift_community_top_k,
                action_concurrency=drift_action_concurrency,
                primer_folds=drift_primer_folds,
                primer_concurrency=drift_primer_concurrency,
            )
        row = self._run(
            "drift",
            drift_executor,
        )
        return {
            "format": "graphrag-drift-mode-benchmark-v1",
            "query": query,
            "settings": {
                "drift_top_k": drift_top_k,
                "drift_profile": drift_profile,
                "drift_max_followups": drift_max_followups,
                "drift_max_depth": drift_max_depth,
                "drift_max_actions_per_depth": drift_max_actions_per_depth,
                "drift_community_top_k": drift_community_top_k,
                "drift_action_concurrency": drift_action_concurrency,
                "drift_primer_folds": drift_primer_folds,
                "drift_primer_concurrency": drift_primer_concurrency,
            },
            "results": [row],
        }

    def run(
        self,
        query: str,
        *,
        local_top_k: int = 5,
        local_community_prop: float = 0.25,
        local_text_unit_prop: float = 0.50,
        global_top_k: int = 5,
        global_map_concurrency: int = 4,
        global_reduce_char_budget: int = 24000,
        drift_top_k: int = 5,
        drift_profile: str = "economy",
        drift_max_followups: int = 2,
        drift_max_depth: int = 2,
        drift_max_actions_per_depth: int = 2,
        drift_community_top_k: int = 1,
        drift_action_concurrency: int = 4,
        drift_primer_folds: int = 5,
        drift_primer_concurrency: int = 5,
    ) -> dict[str, Any]:
        rows = [
            self._run(
                "local",
                lambda: self.local_engine.local(query, top_k=local_top_k),
            ),
            self._run(
                "global",
                lambda: self.global_engine.global_search(
                    query,
                    top_k=global_top_k,
                    map_concurrency=global_map_concurrency,
                    reduce_char_budget=global_reduce_char_budget,
                ),
            ),
            self._run(
                "drift",
                lambda: self.drift_engine.drift(
                    query,
                    top_k=drift_top_k,
                    max_depth=drift_max_depth,
                    max_actions_per_depth=drift_max_actions_per_depth,
                    community_top_k=drift_community_top_k,
                    action_concurrency=drift_action_concurrency,
                    primer_folds=drift_primer_folds,
                    primer_concurrency=drift_primer_concurrency,
                ),
            ),
        ]
        return {
            "format": "graphrag-drift-mode-benchmark-v1",
            "query": query,
            "settings": {
                "local_top_k": local_top_k,
                "local_community_prop": local_community_prop,
                "local_text_unit_prop": local_text_unit_prop,
                "global_top_k": global_top_k,
                "global_map_concurrency": global_map_concurrency,
                "global_reduce_char_budget": global_reduce_char_budget,
                "drift_top_k": drift_top_k,
                "drift_profile": drift_profile,
                "drift_max_followups": drift_max_followups,
                "drift_max_depth": drift_max_depth,
                "drift_max_actions_per_depth": drift_max_actions_per_depth,
                "drift_community_top_k": drift_community_top_k,
                "drift_action_concurrency": drift_action_concurrency,
                "drift_primer_folds": drift_primer_folds,
                "drift_primer_concurrency": drift_primer_concurrency,
            },
            "results": rows,
        }
