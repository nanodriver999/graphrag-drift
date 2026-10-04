from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from graphrag_drift.community_report_generation import (
    CommunityReportCacheBuilder,
    REPORT_FORMAT,
    report_content_hash,
)


@dataclass
class FakeGenerator:
    responses: list[str]
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    dump_sha = "a" * 64
    _write(
        tmp_path / "data/manifest.json",
        {"dump": {"zip_sha256": dump_sha}},
    )
    _write(
        tmp_path / "community_report_inputs/manifest.json",
        {
            "communities": [
                {
                    "community_id": "leiden-8",
                    "context_sha256": "b" * 64,
                    "overview_file": "community_report_inputs/leiden-8/overview.json",
                    "parts": [
                        {
                            "file": "community_report_inputs/leiden-8/evidence_parts/part-000.json"
                        }
                    ],
                }
            ]
        },
    )
    _write(
        tmp_path / "community_report_inputs/leiden-8/overview.json",
        {
            "community": {"properties": {"id": "leiden-8"}},
            "entities": [{"properties": {"id": "업무대행자"}}],
            "relationships": [{"type": "DELEGATES_TO"}],
            "evidence_node_count": 1,
        },
    )
    _write(
        tmp_path / "community_report_inputs/leiden-8/evidence_parts/part-000.json",
        {"evidence_nodes": [{"text": "방사선안전관리 업무를 대행한다."}]},
    )
    return tmp_path


def test_content_hash_matches_documented_formula() -> None:
    dump_sha = "dump"
    context_sha = "context"
    expected = hashlib.sha256(
        f"{dump_sha}\n{context_sha}\n{REPORT_FORMAT}".encode()
    ).hexdigest()
    assert report_content_hash(dump_sha, context_sha) == expected


def test_builder_generates_versioned_report_and_reuses_cache(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    generator = FakeGenerator(
        responses=[
            json.dumps(
                {
                    "summary": "대행 관계 요약",
                    "key_points": ["업무대행자가 일부 안전관리 업무를 대행한다."],
                    "entities": ["업무대행자"],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "title": "방사선안전관리 업무대행",
                    "summary": "업무대행 관계를 설명한다.",
                    "key_findings": ["일부 안전관리 업무를 대행할 수 있다."],
                    "entities": ["업무대행자"],
                    "notes": "근거 1건",
                },
                ensure_ascii=False,
            ),
        ]
    )

    first = CommunityReportCacheBuilder(
        generator=generator,
        generator_name="fake-model",
    ).build(root)

    assert first["generated"] == 1
    assert first["reused"] == 0
    manifest = first["manifest"]
    assert manifest["report_count"] == 1
    report_path = root / manifest["reports"][0]["file"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report"]["title"] == "방사선안전관리 업무대행"
    assert report["source"]["generator"] == "fake-model"

    second = CommunityReportCacheBuilder(
        generator=generator,
        generator_name="fake-model",
    ).build(root)

    assert second["generated"] == 0
    assert second["reused"] == 1
    assert len(generator.prompts) == 2
