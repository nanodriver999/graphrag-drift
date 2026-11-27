"""Fail-closed, date-effective whole-law corpus collector.

This module does not write to Neo4j. A corpus is an immutable directory of
the original official XML, one validated JSON packet per eligible article,
and a manifest that allows byte-for-byte source/provenance verification.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .official_law_api import (
    OfficialLawArticleRequest,
    _article_text,
    _find_text,
)
from .temporal_coverage import normalize_date

API_URL = "https://www.law.go.kr/DRF/lawService.do"
MAX_XML_BYTES = 4 * 1024 * 1024
CORPUS_FORMAT = "graphrag-drift-official-law-corpus-v1"


@dataclass(frozen=True)
class OfficialLawVersionRequest:
    law_id: str
    mst: str
    effective_date: str
    as_of_date: str

    def __post_init__(self) -> None:
        # Share the existing fail-closed law/version/date validation rules.
        OfficialLawArticleRequest(
            law_id=self.law_id,
            mst=self.mst,
            article="1",
            effective_date=self.effective_date,
            as_of_date=self.as_of_date,
        )

    def api_url(self, *, oc: str) -> str:
        if not oc or not oc.strip():
            raise ValueError("LAW_GO_KR_OC authentication is required")
        # Official eflaw docs: omit JO to retrieve every article in one version.
        return API_URL + "?" + urlencode({
            "OC": oc.strip(),
            "target": "eflaw",
            "type": "XML",
            "MST": self.mst,
            "efYd": normalize_date(self.effective_date),
        })


def _article_identifier(node: ET.Element) -> tuple[str, str]:
    raw_number = _find_text(node, "조문번호")
    raw_branch = _find_text(node, "조문가지번호") or "0"
    if not re.fullmatch(r"\d{1,4}", raw_number) or not re.fullmatch(r"\d{1,2}", raw_branch):
        raise ValueError("Official XML contains an invalid or missing article number")
    number, branch = int(raw_number), int(raw_branch)
    if number == 0:
        raise ValueError("Official XML contains a zero article number")
    label = str(number) + (f"-{branch}" if branch else "")
    jo = f"{number:04d}{branch:02d}"
    return label, jo


def parse_official_law_corpus(raw: bytes, request: OfficialLawVersionRequest) -> dict:
    """Validate all article identities, preserve original XML and quarantine future clauses.

    Future-effective articles are *not* accepted as valid 2026 evidence and
    are explicitly enumerated in the manifest instead of silently dropped.
    """
    if not raw or len(raw) > MAX_XML_BYTES:
        raise ValueError("Official law XML is empty or exceeds the 4 MiB limit")
    root = ET.fromstring(raw)
    raw_id = _find_text(root, "법령ID")
    effective = _find_text(root, "시행일자")
    law_name = _find_text(root, "법령명_한글") or _find_text(root, "법령명한글")
    if not raw_id.isdigit() or int(raw_id) != int(request.law_id):
        raise ValueError("Official law ID is absent or does not match request")
    if effective != normalize_date(request.effective_date):
        raise ValueError("Official returned law effective date differs from requested version")
    if not law_name:
        raise ValueError("Official law name is missing")

    nodes = list(root.iter("조문단위"))
    if not nodes:
        raise ValueError("Official law XML has no article nodes")

    raw_sha = hashlib.sha256(raw).hexdigest()
    as_of = normalize_date(request.as_of_date)
    packets: dict[str, dict] = {}
    excluded: list[dict] = []
    seen: set[str] = set()
    for node in nodes:
        article, jo = _article_identifier(node)
        if article in seen:
            raise ValueError(f"Duplicate official article identity: {article}")
        seen.add(article)
        article_date = _find_text(node, "조문시행일자") or effective
        if not re.fullmatch(r"\d{8}", article_date):
            raise ValueError(f"Invalid effective date for article {article}")
        if article_date > as_of:
            excluded.append({"article": article, "effective_date": article_date,
                             "reason": "future_effective"})
            continue

        article_text = _article_text(node)
        if not article_text:
            raise ValueError(f"Empty official text for article {article}")
        packets[article] = {
            "format": "graphrag-drift-official-law-article-v1",
            "source": {
                "provider": "Korean National Law Information Center",
                "public_url": "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=" + request.mst,
                "api_target": "eflaw",
                "mst": request.mst,
                "jo": jo,
                "response_sha256": raw_sha,
            },
            "law_id": request.law_id,
            "law_name": law_name,
            "article": article,
            "effective_date": effective,
            "article_effective_date": article_date,
            "as_of_date": as_of,
            "article_text": article_text,
            "ingestion_status": "downloaded_not_imported",
        }

    if not packets:
        raise ValueError("No valid articles effective on the requested cutoff")
    return {
        "request": request,
        "source_xml_sha256": raw_sha,
        "law_name": law_name,
        "packets": packets,
        "excluded_future_articles": sorted(excluded, key=lambda e: e["article"]),
        "article_count_total": len(nodes),
    }


def fetch_official_law_xml(
    request: OfficialLawVersionRequest, *, oc: str, timeout_seconds: float = 30.0,
) -> bytes:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    # Never print or log the authenticated URL, which includes the OC secret.
    req = Request(request.api_url(oc=oc), headers={
        "User-Agent": "graphrag-drift-law-import/1.0",
    })
    with urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read(MAX_XML_BYTES + 1)
    if not raw or len(raw) > MAX_XML_BYTES:
        raise ValueError("Official law response is empty or exceeds 4 MiB")
    return raw


def _file_for_article(jo: str) -> str:
    return "articles/article-" + jo + ".json"


def _manifest_for(parsed: dict) -> dict:
    req: OfficialLawVersionRequest = parsed["request"]
    return {
        "format": CORPUS_FORMAT,
        "law_id": req.law_id,
        "law_name": parsed["law_name"],
        "mst": req.mst,
        "effective_date": normalize_date(req.effective_date),
        "as_of_date": normalize_date(req.as_of_date),
        "source_xml": "source.xml",
        "source_xml_sha256": parsed["source_xml_sha256"],
        "article_count_total": parsed["article_count_total"],
        "article_count_eligible": len(parsed["packets"]),
        "article_count_excluded_future": len(parsed["excluded_future_articles"]),
        "excluded_future_articles": parsed["excluded_future_articles"],
        "articles": [
            {
                "article": article,
                "effective_date": packet["article_effective_date"],
                "file": _file_for_article(packet["source"]["jo"]),
            }
            for article, packet in sorted(
                parsed["packets"].items(), key=lambda item: int(item[1]["source"]["jo"])
            )
        ],
    }


def write_official_law_corpus(
    root: str | Path, raw: bytes, request: OfficialLawVersionRequest,
    *, required_articles: tuple[str, ...] = (),
) -> dict:
    """Write a new versioned corpus. Never overwrite a prior downloaded corpus."""
    parsed = parse_official_law_corpus(raw, request)
    missing = sorted(set(required_articles) - set(parsed["packets"]))
    if missing:
        raise ValueError(f"Required date-effective articles missing: {missing}")
    manifest = _manifest_for(parsed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "articles").mkdir()
    (root / "source.xml").write_bytes(raw)
    for entry in manifest["articles"]:
        packet = parsed["packets"][entry["article"]]
        (root / entry["file"]).write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    verify_official_law_corpus(root)
    return manifest


def verify_official_law_corpus(root: str | Path) -> dict:
    """Re-parse saved original bytes and compare *all* packets and manifest fields."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != CORPUS_FORMAT:
        raise ValueError("Unknown official law corpus format")
    request = OfficialLawVersionRequest(
        law_id=manifest["law_id"], mst=manifest["mst"],
        effective_date=manifest["effective_date"], as_of_date=manifest["as_of_date"],
    )
    raw = (root / "source.xml").read_bytes()
    parsed = parse_official_law_corpus(raw, request)
    if manifest != _manifest_for(parsed):
        raise ValueError("Official law corpus manifest does not match original XML")
    expected_files = {"manifest.json", "source.xml"}
    for entry in manifest["articles"]:
        path = root / entry["file"]
        if path.is_symlink() or path.parent != root / "articles":
            raise ValueError("Unsafe official corpus article path")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != parsed["packets"][entry["article"]]:
            raise ValueError(f"Official corpus packet differs from XML: {entry['article']}")
        expected_files.add(entry["file"])
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("Official corpus has missing or unexpected files")
    return manifest
