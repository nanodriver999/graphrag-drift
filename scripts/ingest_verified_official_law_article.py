from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from neo4j import GraphDatabase

from graphrag_drift.verified_official_ingestion import (
    import_verified_official_chunk,
    load_and_build_verified_chunk,
)


def _load_base_snapshot(path: Path, dump_path: Path | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "graphrag-drift-neo4j-snapshot-v1":
        raise ValueError("Expected the original community-ready snapshot manifest")
    digest = payload.get("snapshot", {}).get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Snapshot manifest has no SHA-256 identity")
    if dump_path is not None:
        hasher = hashlib.sha256()
        with dump_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise ValueError("Original snapshot dump SHA-256 mismatch")
    return {
        "base_snapshot_sha256": digest,
        "base_snapshot_stage": payload.get("stage"),
    }


def _validate_local_derived_target(uri: str) -> None:
    parsed = urlsplit(uri)
    if parsed.scheme not in {"bolt", "neo4j"} or parsed.hostname not in {
        "localhost", "127.0.0.1", "::1",
    }:
        raise ValueError(
            "Write is permitted only to a locally restored disposable "
            "Neo4j instance; never target Aura or a remote production DB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage or apply one validated official law article to a "
            "disposable derived Neo4j restore, never the original dump."
        )
    )
    parser.add_argument("--packet-json", required=True)
    parser.add_argument("--source-xml", required=True)
    parser.add_argument("--base-snapshot-manifest", required=True)
    parser.add_argument(
        "--base-snapshot-dump",
        help="Original dump whose SHA-256 must match the base manifest for --apply",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write a Chunk to a disposable local Neo4j copy (default: dry run)",
    )
    parser.add_argument(
        "--confirm-disposable-restore", action="store_true",
        help="Explicitly acknowledge target is a disposable restored copy",
    )
    args = parser.parse_args()
    if args.apply and not args.base_snapshot_dump:
        parser.error("--apply requires --base-snapshot-dump for SHA-256 verification")
    props = load_and_build_verified_chunk(args.packet_json, args.source_xml)
    parent = _load_base_snapshot(
        Path(args.base_snapshot_manifest),
        Path(args.base_snapshot_dump) if args.base_snapshot_dump else None,
    )
    report = {
        "format": "graphrag-drift-verified-law-overlay-v1",
        **parent,
        "source_chunk_id": props["sourceChunkId"],
        "article_key": props["articleKey"],
        "official_source_url": props["officialSourceUrl"],
        "source_xml_sha256": props["sourceResponseSha256"],
        "effective_date": props["officialEffectiveDate"],
        "status": "staged_not_imported",
        "derived_snapshot_required": True,
    }
    if args.apply:
        if not args.confirm_disposable_restore:
            parser.error(
                "--apply requires --confirm-disposable-restore; "
                "restore the existing dump into a separate local Neo4j first"
            )
        uri = os.getenv("LOCAL_NEO4J_URI", "bolt://localhost:17687")
        _validate_local_derived_target(uri)
        user = os.getenv("LOCAL_NEO4J_USERNAME", "neo4j")
        password = os.getenv("LOCAL_NEO4J_PASSWORD")
        database = os.getenv("LOCAL_NEO4J_DATABASE", "neo4j")
        if not password:
            parser.error("LOCAL_NEO4J_PASSWORD is required for --apply")
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            written = import_verified_official_chunk(
                driver, props, database=database
            )
        report["status"] = written["status"]
        report["source_chunk_id"] = written["source_chunk_id"]

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Official law overlay {report['status']}: "
        f"{report['article_key']}. "
        "Original source snapshot is not modified; generate a new "
        "Neo4j snapshot with its own SHA-256 after applying."
    )


if __name__ == "__main__":
    main()
