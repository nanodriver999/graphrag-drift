from __future__ import annotations

import json

import pytest
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



def _dated_store(root: Path) -> FileCommunityReportStore:
    root_rel = Path("community_reports/dated")
    items = [
        ("current", "20260520", False),
        ("future", "20270220", False),
        ("undated", None, True),
    ]
    rows = []
    for community_id, effective, unknown in items:
        provenance = None
        if effective is not None:
            article_key = (
                f"kr:law:011435:mst:1:effective:{effective}:art:53-3"
            )
            provenance = {
                "format": "graphrag-drift-community-source-provenance-v1",
                "source_article_versions": [{
                    "article_key": article_key,
                    "law_id": "011435",
                    "article": "53-3",
                    "effective_date": effective,
                }],
                "source_chunk_count": 1,
                "undated_or_unversioned_chunk_count": 0,
                "other_evidence_count": 0,
                "entity_description_count": 0,
            }
        report_rel = root_rel / f"community-{community_id}.json"
        _write_json(root / report_rel, {
            "community_id": community_id,
            "source_provenance": provenance,
            "report": {
                "title": f"{community_id} radiation",
                "summary": "radiation safety management",
                "key_findings": ["radiation safety"],
            },
        })
        rows.append({"community_id": community_id, "file": report_rel.as_posix()})
    manifest = root_rel / "manifest.json"
    _write_json(root / manifest, {"reports": rows})
    return FileCommunityReportStore.from_relative_manifest(root, manifest)


def test_cached_reports_date_filter_excludes_future_source_context(tmp_path: Path) -> None:
    retriever = CachedCommunityReportRetriever(
        local_retriever=InMemoryRetriever(hits=[], reports=[]),
        report_store=_dated_store(tmp_path),
        as_of_date="2026-10-05",
        temporal_policy="exclude_future",
    )
    selected = retriever.community_reports("radiation safety", top_k=10)
    assert {r.id for r in selected} == {"current", "undated"}


def test_cached_reports_strict_date_policy_excludes_uncertified_reports(tmp_path: Path) -> None:
    retriever = CachedCommunityReportRetriever(
        local_retriever=InMemoryRetriever(hits=[], reports=[]),
        report_store=_dated_store(tmp_path),
        as_of_date="2026-10-05",
        temporal_policy="require_verified",
    )
    selected = retriever.community_reports("radiation safety", top_k=10)
    assert [r.id for r in selected] == ["current"]


def test_cached_reports_default_keeps_historical_behavior(tmp_path: Path) -> None:
    retriever = CachedCommunityReportRetriever(
        local_retriever=InMemoryRetriever(hits=[], reports=[]),
        report_store=_dated_store(tmp_path),
    )
    selected = retriever.community_reports("radiation safety", top_k=10)
    assert {r.id for r in selected} == {"current", "future", "undated"}


@pytest.mark.parametrize("policy", ["exclude_future", "require_verified"])
def test_temporal_policy_requires_date(policy: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="as_of_date"):
        CachedCommunityReportRetriever(
            local_retriever=InMemoryRetriever(hits=[], reports=[]),
            report_store=_dated_store(tmp_path),
            temporal_policy=policy,
        )
