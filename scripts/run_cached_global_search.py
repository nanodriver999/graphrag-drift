#!/usr/bin/env python3
"""Run GraphRAG Global Search using cached Community Reports from the data repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_drift import (
    CachedCommunityReportRetriever,
    FileCommunityReportStore,
    GraphRAGEngine,
    InMemoryRetriever,
)


def resolve_manifest(data_repo: Path) -> Path:
    source_manifest = json.loads(
        (data_repo / "data" / "manifest.json").read_text(encoding="utf-8")
    )
    dump_sha = source_manifest["dump"]["zip_sha256"]
    return Path("community_reports") / dump_sha[:12] / "manifest.json"


def run_cached_global_search(
    data_repo: Path,
    query: str,
    *,
    top_k: int = 5,
    manifest: Path | None = None,
) -> dict:
    relative_manifest = manifest or resolve_manifest(data_repo)

    store = FileCommunityReportStore.from_relative_manifest(
        data_repo,
        relative_manifest,
    )
    retriever = CachedCommunityReportRetriever(
        local_retriever=InMemoryRetriever(hits=[], reports=[]),
        report_store=store,
    )
    engine = GraphRAGEngine(retriever=retriever)
    return engine.global_search(query, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument(
        "--data-repo",
        required=True,
        type=Path,
        help="Path to a checkout of nanodriver999/graphrag-drift-data",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional report manifest path relative to --data-repo",
    )
    args = parser.parse_args()

    result = run_cached_global_search(
        args.data_repo,
        args.query,
        top_k=args.top_k,
        manifest=args.manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
