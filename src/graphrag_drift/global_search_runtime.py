from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .cached_retrieval import CachedCommunityReportRetriever
from .community_report_store import FileCommunityReportStore
from .core import GraphRAGEngine
from .global_llm_reasoner import GlobalLLMReasoner
from .openai_compatible import OpenAICompatibleTextGenerator
from .retrieval import InMemoryRetriever


@dataclass(frozen=True)
class GlobalSearchLLMSettings:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "GlobalSearchLLMSettings":
        base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "").strip()
        model = os.environ.get("OPENAI_COMPATIBLE_MODEL", "").strip()
        api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        timeout = float(os.environ.get("OPENAI_COMPATIBLE_TIMEOUT_SECONDS", "60"))
        temperature = float(os.environ.get("OPENAI_COMPATIBLE_TEMPERATURE", "0"))

        if not base_url:
            raise ValueError("OPENAI_COMPATIBLE_BASE_URL is required")
        if not model:
            raise ValueError("OPENAI_COMPATIBLE_MODEL is required")

        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout,
            temperature=temperature,
        )


def resolve_current_report_manifest(data_repo: str | Path) -> Path:
    root = Path(data_repo)
    source_manifest = json.loads(
        (root / "data" / "manifest.json").read_text(encoding="utf-8")
    )
    dump_sha = source_manifest["dump"]["zip_sha256"]
    return Path("community_reports") / dump_sha[:12] / "manifest.json"


def build_llm_cached_global_search_engine(
    data_repo: str | Path,
    settings: GlobalSearchLLMSettings,
) -> GraphRAGEngine:
    root = Path(data_repo)
    report_store = FileCommunityReportStore.from_relative_manifest(
        root,
        resolve_current_report_manifest(root),
    )
    retriever = CachedCommunityReportRetriever(
        local_retriever=InMemoryRetriever(hits=[], reports=[]),
        report_store=report_store,
    )
    generator = OpenAICompatibleTextGenerator(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
        temperature=settings.temperature,
    )
    reasoner = GlobalLLMReasoner(generator=generator)
    return GraphRAGEngine(retriever=retriever, reasoner=reasoner)
