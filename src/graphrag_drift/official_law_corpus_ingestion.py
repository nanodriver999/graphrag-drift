"""All-or-nothing overlay of an independently verified whole-law corpus.

The input corpus is verified against its original official XML before any
transaction, and only disposable local Neo4j restores may be modified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .official_law_corpus import verify_official_law_corpus
from .verified_official_ingestion import (
    _apply_in_transaction,
    build_verified_official_chunk,
)


def prepare_official_corpus_chunks(
    corpus_root: str | Path, *, required_articles: tuple[str, ...] = (),
) -> tuple[dict, list[dict[str, Any]]]:
    """Fail closed on tampering or missing as-of-date evidence."""
    root = Path(corpus_root)
    manifest = verify_official_law_corpus(root)
    raw_xml = (root / "source.xml").read_bytes()
    chunks: list[dict[str, Any]] = []
    found: set[str] = set()
    identities: set[str] = set()
    for item in manifest["articles"]:
        packet = json.loads((root / item["file"]).read_text(encoding="utf-8"))
        properties = build_verified_official_chunk(packet, raw_xml)
        if properties["articleKey"] in identities:
            raise ValueError("Duplicate versioned articleKey in official corpus")
        identities.add(properties["articleKey"])
        found.add(packet["article"])
        chunks.append(properties)
    absent = sorted(set(required_articles) - found)
    if absent:
        raise ValueError(f"Required date-effective articles missing: {absent}")
    if not chunks:
        raise ValueError("Official corpus is empty")
    return manifest, chunks


def _apply_all_in_transaction(tx: Any, chunks: list[dict[str, Any]]) -> dict:
    """One Neo4j transaction for all articles, avoiding partial batch writes."""
    created = 0
    already_present = 0
    for properties in chunks:
        result = _apply_in_transaction(tx, properties)
        if result["status"] == "created":
            created += 1
        elif result["status"] == "already_present":
            already_present += 1
        else:
            raise RuntimeError("Unexpected source import status")
    return {
        "status": "applied_to_disposable_restore",
        "created": created,
        "already_present": already_present,
        "total": len(chunks),
    }


def import_verified_official_corpus(
    driver: Any, chunks: list[dict[str, Any]], *, database: str,
) -> dict:
    if not database:
        raise ValueError("Explicit Neo4j database name is required")
    if not chunks:
        raise ValueError("No verified articles to import")
    identities: set[str] = set()
    for props in chunks:
        if props.get("sourceStatus") != "verified_official_xml":
            raise ValueError("Cannot import unverified article")
        article_key = props.get("articleKey")
        if not isinstance(article_key, str) or article_key in identities:
            raise ValueError("Duplicate or missing articleKey")
        identities.add(article_key)
    with driver.session(database=database) as session:
        return session.execute_write(_apply_all_in_transaction, chunks)


def build_official_corpus_overlay_report(
    corpus_manifest: dict, parent_manifest: dict, *, result: dict | None = None,
) -> dict:
    """Label an overlay as derived/staged, never as a finalized rebuilt snapshot."""
    return {
        "format": "graphrag-drift-official-law-corpus-overlay-v1",
        **parent_manifest,
        "official_law_id": corpus_manifest["law_id"],
        "official_mst": corpus_manifest["mst"],
        "official_effective_date": corpus_manifest["effective_date"],
        "as_of_date": corpus_manifest["as_of_date"],
        "official_source_xml_sha256": corpus_manifest["source_xml_sha256"],
        "eligible_article_count": corpus_manifest["article_count_eligible"],
        "excluded_future_article_count": corpus_manifest["article_count_excluded_future"],
        "status": "staged_not_imported" if result is None else result["status"],
        "derived_snapshot_required": True,
        "graph_edges_and_communities_rebuilt": False,
        "import_summary": result,
    }
