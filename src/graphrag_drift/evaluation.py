from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Any

from .core import GraphRAGEngine
from .global_llm_reasoner import TextGenerator


@dataclass
class InstrumentedTextGenerator:
    generator: TextGenerator
    call_count: int = 0
    total_seconds: float = 0.0
    prompts: list[str] = field(default_factory=list)
    _lock: Any = field(default_factory=Lock, init=False, repr=False)

    def reset(self) -> None:
        with self._lock:
            self.call_count = 0
            self.total_seconds = 0.0
            self.prompts.clear()

    def generate(self, prompt: str) -> str:
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
        if mode == "drift":
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
        return metrics

    def _run(self, mode: str, fn) -> dict[str, Any]:
        self.generator_probe.reset()
        start = perf_counter()
        result = fn()
        elapsed = perf_counter() - start
        return self._metrics(mode, result, elapsed)

    def run(
        self,
        query: str,
        *,
        local_top_k: int = 5,
        global_top_k: int = 5,
        drift_top_k: int = 5,
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
                lambda: self.global_engine.global_search(query, top_k=global_top_k),
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
                "global_top_k": global_top_k,
                "drift_top_k": drift_top_k,
                "drift_max_depth": drift_max_depth,
                "drift_max_actions_per_depth": drift_max_actions_per_depth,
                "drift_community_top_k": drift_community_top_k,
                "drift_action_concurrency": drift_action_concurrency,
                "drift_primer_folds": drift_primer_folds,
                "drift_primer_concurrency": drift_primer_concurrency,
            },
            "results": rows,
        }
