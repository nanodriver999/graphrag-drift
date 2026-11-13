from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from .official_law_api import (
    OfficialLawArticleRequest,
    parse_official_article_xml,
)

ARTICLE_DUPLICATE_QUERY = """
MATCH (chunk:Chunk)
WHERE chunk.articleKey = $article_key
RETURN chunk.sourceChunkId AS source_id,
       chunk.text AS text,
       chunk.sourceResponseSha256 AS response_sha256
ORDER BY source_id
LIMIT 100
"""

CREATE_OFFICIAL_CHUNK_QUERY = """
CREATE (chunk:Chunk)
SET chunk += $properties
RETURN chunk.sourceChunkId AS source_id
"""


def build_verified_official_chunk(
    packet: dict[str, Any], raw_xml: bytes,
) -> dict[str, Any]:
    """Build a Chunk from an independently verified official source packet.

    The original XML must match every field of the packet, including its
    SHA-256. A JSON file alone is not proof of authenticated legal content.
    """
    if not isinstance(packet, dict):
        raise ValueError("Official source packet must be an object")
    if packet.get("format") != "graphrag-drift-official-law-article-v1":
        raise ValueError("Unknown official law packet format")
    if packet.get("ingestion_status") != "downloaded_not_imported":
        raise ValueError("Source packet is not in the verified staging state")
    source = packet.get("source")
    if not isinstance(source, dict) or source.get("api_target") != "eflaw":
        raise ValueError("Official source provenance is missing or wrong")
    required = ("law_id", "article", "effective_date", "as_of_date")
    if any(not isinstance(packet.get(k), str) for k in required):
        raise ValueError("Incomplete official law version information")
    mst = source.get("mst")
    if not isinstance(mst, str):
        raise ValueError("Official law MST must be a string")
    request = OfficialLawArticleRequest(
        law_id=packet["law_id"],
        mst=mst,
        article=packet["article"],
        effective_date=packet["effective_date"],
        as_of_date=packet["as_of_date"],
    )
    parsed = parse_official_article_xml(raw_xml, request)
    if parsed != packet:
        raise ValueError(
            "Original official XML and submitted packet do not match; "
            "refusing to import altered or stale legal material"
        )
    if not packet["law_name"].strip() or not packet["article_text"].strip():
        raise ValueError("Official law name or article text is empty")
    law_id = request.law_id.zfill(6)
    law_version_key = (
        f"kr:law:{law_id}:mst:{mst}:effective:"
        f"{parsed['effective_date']}"
    )
    article_key = law_version_key + ":art:" + request.article
    number, _, branch = request.article.partition("-")
    article_label = f"제{int(number)}조" + (
        f"의{int(branch)}" if branch else ""
    )
    chunk_id = (
        f"official-eflaw:{law_id}:{mst}:"
        f"{parsed['effective_date']}:{request.article}"
    )
    return {
        "id": chunk_id,
        "sourceChunkId": chunk_id,
        "lawId": law_id,
        "lawTitle": parsed["law_name"],
        "lawVersionKey": law_version_key,
        "articleKey": article_key,
        "article": article_label,
        "officialEffectiveDate": parsed["effective_date"],
        "articleEffectiveDate": parsed["article_effective_date"],
        "text": parsed["law_name"] + " " + parsed["article_text"],
        "position": 0,
        "sourceType": "body",
        "sourcePointer": f"/official-api/eflaw/{mst}/article/{request.jo}",
        "officialSourceUrl": source["public_url"],
        "sourceResponseSha256": source["response_sha256"],
        "sourceStatus": "verified_official_xml",
        "sourceSchemaVersion": "verified-official-article-v1",
    }


def _apply_in_transaction(tx: Any, properties: dict[str, Any]) -> dict[str, Any]:
    rows = list(tx.run(
        ARTICLE_DUPLICATE_QUERY,
        article_key=properties["articleKey"],
    ))
    if rows:
        if len(rows) == 1 and all(
            rows[0].get(column) == properties[prop]
            for column, prop in (
                ("source_id", "sourceChunkId"),
                ("text", "text"),
                ("response_sha256", "sourceResponseSha256"),
            )
        ):
            return {
                "status": "already_present",
                "source_chunk_id": properties["sourceChunkId"],
            }
        raise ValueError(
            "An existing Chunk uses the same versioned articleKey. "
            "Manual source reconciliation is required; refusing duplicates."
        )
    created = tx.run(
        CREATE_OFFICIAL_CHUNK_QUERY, properties=properties,
    ).single()
    if not created or created.get("source_id") != properties["sourceChunkId"]:
        raise RuntimeError("Neo4j did not confirm official Chunk creation")
    return {
        "status": "created",
        "source_chunk_id": properties["sourceChunkId"],
    }


def import_verified_official_chunk(
    driver: Any,
    properties: dict[str, Any],
    *,
    database: str,
) -> dict[str, Any]:
    """Add one official Chunk transactionally to a disposable Neo4j copy."""
    if not database:
        raise ValueError("Explicit Neo4j database name is required")
    if properties.get("sourceStatus") != "verified_official_xml":
        raise ValueError("Refusing to import an unverified Chunk")
    with driver.session(database=database) as session:
        return session.execute_write(_apply_in_transaction, properties)


def load_and_build_verified_chunk(
    packet_path: str | Path, xml_path: str | Path,
) -> dict[str, Any]:
    raw = Path(xml_path).read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Source XML must not exceed 4 MiB")
    packet = json.loads(Path(packet_path).read_text(encoding="utf-8"))
    return build_verified_official_chunk(packet, raw)
