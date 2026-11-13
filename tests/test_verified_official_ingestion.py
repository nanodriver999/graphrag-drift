from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from graphrag_drift.official_law_api import (
    OfficialLawArticleRequest,
    parse_official_article_xml,
)
from graphrag_drift.verified_official_ingestion import (
    ARTICLE_DUPLICATE_QUERY,
    CREATE_OFFICIAL_CHUNK_QUERY,
    build_verified_official_chunk,
    import_verified_official_chunk,
    load_and_build_verified_chunk,
)
from scripts.ingest_verified_official_law_article import (
    _load_base_snapshot,
    _validate_local_derived_target,
)


XML = """<?xml version="1.0" encoding="utf-8"?>
<법령><기본정보>
<법령ID>11435</법령ID>
<법령명_한글>원자력안전법</법령명_한글>
<시행일자>20260519</시행일자>
</기본정보>
<조문><조문단위>
<조문번호>53</조문번호><조문가지번호>3</조문가지번호>
<조문시행일자>20260519</조문시행일자>
<조문내용>제53조의3(방사선안전관리자)</조문내용>
<항><항내용>① 안전관리규정 및 기술기준 확인</항내용></항>
<호><호내용>1. 안전관리규정 점검</호내용></호>
</조문단위></조문>
</법령>
""".encode("utf-8")


def _verified_packet() -> dict:
    request = OfficialLawArticleRequest(
        law_id="011435", mst="286119", article="53-3",
        effective_date="2026-05-19", as_of_date="2026-10-05",
    )
    return parse_official_article_xml(XML, request)


def test_staged_official_chunk_preserves_exact_law_version_and_source() -> None:
    chunk = build_verified_official_chunk(_verified_packet(), XML)
    assert chunk["articleKey"] == (
        "kr:law:011435:mst:286119:effective:20260519:art:53-3"
    )
    assert chunk["lawVersionKey"] == (
        "kr:law:011435:mst:286119:effective:20260519"
    )
    assert chunk["officialEffectiveDate"] == "20260519"
    assert chunk["articleEffectiveDate"] == "20260519"
    assert chunk["sourceChunkId"].startswith("official-eflaw:")
    assert chunk["sourceResponseSha256"] == _verified_packet()["source"]["response_sha256"]
    assert chunk["sourceType"] == "body"
    assert chunk["sourceStatus"] == "verified_official_xml"
    assert "제53조의3" in chunk["text"]
    assert "기술기준 확인" in chunk["text"]


@pytest.mark.parametrize("change", [
    ("article_text", "제53조의3 가공된 의무"),
    ("effective_date", "20260901"),
    ("law_name", "다른 법령"),
    ("article_effective_date", "20260101"),
    ("ingestion_status", "already_imported"),
])
def test_tampered_or_unstaged_packet_rejected(change: tuple[str, str]) -> None:
    packet = _verified_packet()
    packet[change[0]] = change[1]
    with pytest.raises(ValueError):
        build_verified_official_chunk(packet, XML)


def test_wrong_raw_xml_sha_or_article_rejected() -> None:
    packet = _verified_packet()
    with pytest.raises(ValueError):
        build_verified_official_chunk(packet, XML.replace(b"11435", b"11434"))


def test_verified_packet_can_be_loaded_from_separate_artifacts(tmp_path: Path) -> None:
    packet_path = tmp_path / "official.json"
    xml_path = tmp_path / "official.xml"
    packet_path.write_text(json.dumps(_verified_packet(), ensure_ascii=False))
    xml_path.write_bytes(XML)
    chunk = load_and_build_verified_chunk(packet_path, xml_path)
    assert chunk["officialSourceUrl"] == (
        "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=286119"
    )


class FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None


class FakeTx:
    def __init__(self, preexisting: list[dict] | None = None) -> None:
        self.preexisting = preexisting or []
        self.creates: list[dict] = []

    def run(self, query: str, **params):
        if query == ARTICLE_DUPLICATE_QUERY:
            assert params["article_key"].endswith(":art:53-3")
            return FakeCursor(self.preexisting)
        assert query == CREATE_OFFICIAL_CHUNK_QUERY
        props = params["properties"]
        self.creates.append(props)
        return FakeCursor([{"source_id": props["sourceChunkId"]}])


class FakeSession:
    def __init__(self, tx: FakeTx, database: str) -> None:
        self.tx = tx
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_write(self, func, properties):
        return func(self.tx, properties)


class FakeDriver:
    def __init__(self, tx: FakeTx) -> None:
        self.tx = tx
        self.called_db: str | None = None

    def session(self, *, database: str):
        self.called_db = database
        return FakeSession(self.tx, database)


def test_first_import_creates_one_source_chunk_in_single_write_transaction() -> None:
    tx = FakeTx()
    driver = FakeDriver(tx)
    chunk = build_verified_official_chunk(_verified_packet(), XML)
    result = import_verified_official_chunk(driver, chunk, database="derived")
    assert result["status"] == "created"
    assert len(tx.creates) == 1
    assert driver.called_db == "derived"


def test_idempotent_exact_source_does_not_duplicate_chunk() -> None:
    chunk = build_verified_official_chunk(_verified_packet(), XML)
    tx = FakeTx([{
        "source_id": chunk["sourceChunkId"],
        "text": chunk["text"],
        "response_sha256": chunk["sourceResponseSha256"],
    }])
    result = import_verified_official_chunk(FakeDriver(tx), chunk, database="neo4j")
    assert result["status"] == "already_present"
    assert tx.creates == []


def test_conflicting_article_key_must_not_be_overwritten() -> None:
    chunk = build_verified_official_chunk(_verified_packet(), XML)
    tx = FakeTx([{
        "source_id": "untrusted-fragment",
        "text": "different law text",
        "response_sha256": "0" * 64,
    }])
    with pytest.raises(ValueError, match="Manual source reconciliation"):
        import_verified_official_chunk(FakeDriver(tx), chunk, database="neo4j")
    assert tx.creates == []


def test_undated_or_remote_write_target_rejected() -> None:
    for target in (
        "neo4j+s://my-aura.databases.neo4j.io",
        "bolt://public-server:7687",
        "https://www.law.go.kr",
    ):
        with pytest.raises(ValueError, match="locally restored"):
            _validate_local_derived_target(target)
    _validate_local_derived_target("bolt://127.0.0.1:17687")


def test_snapshot_manifest_must_preserve_original_sha256(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": "graphrag-drift-neo4j-snapshot-v1",
        "stage": "community-ready",
        "snapshot": {"sha256": "a" * 64},
    }))
    row = _load_base_snapshot(manifest)
    assert row["base_snapshot_sha256"] == "a" * 64
    assert row["base_snapshot_stage"] == "community-ready"
