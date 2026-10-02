from __future__ import annotations

import json
from pathlib import Path

from graphrag_drift.cached_retrieval import CachedCommunityReportRetriever
from graphrag_drift.community_report_store import FileCommunityReportStore
from graphrag_drift.models import CommunityReport, SearchHit
from graphrag_drift.retrieval import InMemoryRetriever


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_store(root: Path) -> FileCommunityReportStore:
    report_dir = root / "community_reports" / "abc123"
    rows = []
    for community_id, title, summary, finding in [
        (
            "leiden-4",
            "Operator safety duties",
            "Nuclear operators manage radiation protection and incident reporting.",
            "Dose management is required.",
        ),
        (
            "leiden-8",
            "Radiation safety delegation",
            "Licensed users may delegate radiation safety work to registered agents.",
            "A radiation safety manager must be appointed.",
        ),
    ]:
        path = report_dir / f"community-{community_id}.json"
        _write_json(
            path,
            {
                "community_id": community_id,
                "report": {
                    "title": title,
                    "summary": summary,
                    "key_findings": [finding],
                },
            },
        )
        rows.append(
            {
                "community_id": community_id,
                "file": str(path.relative_to(root)),
            }
        )

    manifest = report_dir / "manifest.json"
    _write_json(manifest, {"reports": rows})
    return FileCommunityReportStore.from_relative_manifest(
        root,
        manifest.relative_to(root),
    )


def test_cached_retriever_delegates_local_search(tmp_path: Path) -> None:
    local = InMemoryRetriever(
        hits=[SearchHit(id="local-1", text="local evidence")],
        reports=[CommunityReport(id="ignored", summary="ignored")],
    )
    retriever = CachedCommunityReportRetriever(
        local_retriever=local,
        report_store=_build_store(tmp_path),
    )

    hits = retriever.local_search("local", top_k=1)

    assert [hit.id for hit in hits] == ["local-1"]


def test_cached_retriever_ranks_cached_reports_for_global_search(tmp_path: Path) -> None:
    local = InMemoryRetriever(hits=[], reports=[])
    retriever = CachedCommunityReportRetriever(
        local_retriever=local,
        report_store=_build_store(tmp_path),
    )

    reports = retriever.community_reports("radiation safety manager", top_k=1)

    assert len(reports) == 1
    assert reports[0].id == "leiden-8"
    assert "Radiation safety delegation" in reports[0].summary
