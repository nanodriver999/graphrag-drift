from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag_drift.global_search_runtime import (
    GlobalSearchLLMSettings,
    build_llm_cached_global_search_engine,
    resolve_current_report_manifest,
)
from graphrag_drift.global_llm_reasoner import GlobalLLMReasoner
from graphrag_drift.openai_compatible import OpenAICompatibleTextGenerator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_global_search_llm_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "local-model")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_COMPATIBLE_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("OPENAI_COMPATIBLE_TEMPERATURE", "0.2")

    settings = GlobalSearchLLMSettings.from_env()

    assert settings.base_url == "http://localhost:8000"
    assert settings.model == "local-model"
    assert settings.api_key == "secret"
    assert settings.timeout_seconds == 15
    assert settings.temperature == 0.2


def test_global_search_llm_settings_requires_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_MODEL", raising=False)

    with pytest.raises(ValueError, match="BASE_URL"):
        GlobalSearchLLMSettings.from_env()


def test_resolve_current_report_manifest_uses_dump_prefix(tmp_path: Path) -> None:
    dump_sha = "b" * 64
    _write_json(tmp_path / "data" / "manifest.json", {"dump": {"zip_sha256": dump_sha}})

    path = resolve_current_report_manifest(tmp_path)

    assert path == Path("community_reports") / dump_sha[:12] / "manifest.json"


def test_build_llm_cached_global_search_engine_composes_existing_parts(tmp_path: Path) -> None:
    dump_sha = "c" * 64
    prefix = dump_sha[:12]
    report_dir = tmp_path / "community_reports" / prefix
    report_file = report_dir / "community-leiden-8.json"

    _write_json(tmp_path / "data" / "manifest.json", {"dump": {"zip_sha256": dump_sha}})
    _write_json(
        report_file,
        {
            "community_id": "leiden-8",
            "report": {
                "title": "Radiation safety",
                "summary": "A safety manager is appointed.",
                "key_findings": ["Delegation is regulated."],
            },
        },
    )
    _write_json(
        report_dir / "manifest.json",
        {
            "reports": [
                {
                    "community_id": "leiden-8",
                    "file": str(report_file.relative_to(tmp_path)),
                }
            ]
        },
    )

    engine = build_llm_cached_global_search_engine(
        tmp_path,
        GlobalSearchLLMSettings(
            base_url="http://localhost:8000",
            model="local-model",
        ),
    )

    assert isinstance(engine.reasoner, GlobalLLMReasoner)
    assert isinstance(engine.reasoner.generator, OpenAICompatibleTextGenerator)
    assert engine.reasoner.generator.model == "local-model"
    reports = engine.retriever.community_reports("safety manager", top_k=1)
    assert [report.id for report in reports] == ["leiden-8"]
