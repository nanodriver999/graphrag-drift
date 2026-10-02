from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cached_global_search_cli_uses_current_dump_prefix(tmp_path: Path) -> None:
    dump_sha = "a" * 64
    prefix = dump_sha[:12]

    _write_json(
        tmp_path / "data" / "manifest.json",
        {"dump": {"zip_sha256": dump_sha}},
    )

    report_dir = tmp_path / "community_reports" / prefix
    report_file = report_dir / "community-leiden-8.json"
    _write_json(
        report_file,
        {
            "community_id": "leiden-8",
            "report": {
                "title": "Radiation safety delegation",
                "summary": "Licensed users appoint a radiation safety manager.",
                "key_findings": [
                    "Registered agents can perform delegated safety work."
                ],
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

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_cached_global_search.py",
            "radiation safety manager",
            "--data-repo",
            str(tmp_path),
            "--top-k",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)

    assert len(result["partials"]) == 1
    assert "leiden-8" in result["partials"][0][0]
    assert "Radiation safety delegation" in result["answer"]
