from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import CommunityReport


@dataclass(frozen=True)
class FileCommunityReportStore:
    """Load versioned CommunityReport cache files from a data repository checkout.

    The data repository manifest stores report paths relative to its repository root.
    This adapter intentionally only loads/validates cached reports; retrieval and
    ranking remain separate concerns.
    """

    repository_root: Path
    manifest_path: Path

    @classmethod
    def from_relative_manifest(
        cls,
        repository_root: str | Path,
        manifest_path: str | Path,
    ) -> "FileCommunityReportStore":
        root = Path(repository_root)
        return cls(repository_root=root, manifest_path=root / Path(manifest_path))

    def _load_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self) -> list[CommunityReport]:
        manifest = self._load_json(self.manifest_path)
        rows = manifest.get("reports")
        if not isinstance(rows, list):
            raise ValueError("Community Report manifest must contain a reports list")

        reports: list[CommunityReport] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Community Report manifest entries must be objects")

            relative_file = row.get("file")
            if not isinstance(relative_file, str) or not relative_file:
                raise ValueError("Community Report manifest entry is missing file")

            report_path = self.repository_root / relative_file
            payload = self._load_json(report_path)

            community_id = payload.get("community_id")
            report_body = payload.get("report")
            if not isinstance(community_id, (str, int)):
                raise ValueError(f"Cached report is missing community_id: {report_path}")
            if not isinstance(report_body, dict):
                raise ValueError(f"Cached report is missing report body: {report_path}")

            title = report_body.get("title")
            summary = report_body.get("summary")
            findings = report_body.get("key_findings", [])
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"Cached report is missing title: {report_path}")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(f"Cached report is missing summary: {report_path}")
            if not isinstance(findings, list) or any(
                not isinstance(item, str) for item in findings
            ):
                raise ValueError(f"Cached report key_findings must be strings: {report_path}")

            text_parts = [title.strip(), summary.strip()]
            text_parts.extend(item.strip() for item in findings if item.strip())

            reports.append(
                CommunityReport(
                    id=str(community_id),
                    summary="\n".join(text_parts),
                    score=1.0,
                )
            )

        return reports
