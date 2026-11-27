from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neo4j import GraphDatabase

from graphrag_drift.official_law_corpus_ingestion import (
    build_official_corpus_overlay_report,
    import_verified_official_corpus,
    prepare_official_corpus_chunks,
)
from graphrag_drift.verified_official_ingestion import (
    load_base_snapshot,
    validate_local_derived_target,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a whole-law corpus and stage or atomically overlay its "
            "articles into a disposable local Neo4j restore only."
        )
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--base-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--base-snapshot-dump", type=Path,
                        help="Original immutable neo4j.dump; mandatory for --apply")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--require-article", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-disposable-restore", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.confirm_disposable_restore:
        parser.error("--apply requires --confirm-disposable-restore")
    if args.apply and args.base_snapshot_dump is None:
        parser.error("--apply requires --base-snapshot-dump for SHA-256 verification")

    manifest, chunks = prepare_official_corpus_chunks(
        args.corpus_dir, required_articles=tuple(args.require_article)
    )
    base = load_base_snapshot(
        args.base_snapshot_manifest, args.base_snapshot_dump
    )
    result = None
    if args.apply:
        uri = os.environ.get("LOCAL_NEO4J_URI", "bolt://localhost:17687")
        validate_local_derived_target(uri)
        password = os.environ.get("LOCAL_NEO4J_PASSWORD")
        if not password:
            parser.error("LOCAL_NEO4J_PASSWORD required for --apply")
        username = os.environ.get("LOCAL_NEO4J_USERNAME", "neo4j")
        database = os.environ.get("LOCAL_NEO4J_DATABASE", "neo4j")
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            driver.verify_connectivity()
            result = import_verified_official_corpus(driver, chunks, database=database)
    report = build_official_corpus_overlay_report(manifest, base, result=result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
