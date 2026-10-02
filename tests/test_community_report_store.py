from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag_drift.community_report_store import FileCommunityReportStore


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_file_community_report_store_loads_cached_reports(tmp_path: Path) -> None:
    report_rel = "community_reports/abc123/community-leiden-4.json"
    manifest_rel = "community_reports/abc123/manifest.json"

    _write_json(
        tmp_path / report_rel,
        {
            "community_id": "leiden-4",
            "report": {
                "title": "Operator duties",
                "summary": "Operators have continuing safety obligations.",
                "key_findings": [
                    "Dose management is required.",
                    "Incidents require reporting.",
                ],
            },
        },
    )
    _write_json(
        tmp_path / manifest_rel,
        {
            "reports": [
                {
                    "community_id": "leiden-4",
                    "file": report_rel,
                }
            ]
        },
    )

    store = FileCommunityReportStore.from_relative_manifest(
        tmp_path,
        manifest_rel,
    )

    reports = store.load()

    assert len(reports) == 1
    assert reports[0].id == "leiden-4"
    assert "Operator duties" in reports[0].summary
    assert "Dose management is required." in reports[0].summary


def test_file_community_report_store_rejects_missing_report_body(tmp_path: Path) -> None:
    report_rel = "community_reports/abc123/community-bad.json"
    manifest_rel = "community_reports/abc123/manifest.json"

    _write_json(tmp_path / report_rel, {"community_id": "bad"})
    _write_json(tmp_path / manifest_rel, {"reports": [{"file": report_rel}]})

    store = FileCommunityReportStore.from_relative_manifest(tmp_path, manifest_rel)

    with pytest.raises(ValueError, match="report body"):
        store.load()
