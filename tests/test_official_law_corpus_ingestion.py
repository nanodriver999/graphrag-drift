from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag_drift.official_law_corpus import (
    OfficialLawVersionRequest,
    write_official_law_corpus,
)
from graphrag_drift.official_law_corpus_ingestion import (
    build_official_corpus_overlay_report,
    import_verified_official_corpus,
    prepare_official_corpus_chunks,
)
from graphrag_drift.verified_official_ingestion import (
    ARTICLE_DUPLICATE_QUERY, CREATE_OFFICIAL_CHUNK_QUERY,
)

XML = """<?xml version="1.0" encoding="utf-8"?>
<법령><기본정보><법령ID>11435</법령ID>
<법령명_한글>원자력안전법</법령명_한글><시행일자>20260519</시행일자>
</기본정보><조문>
<조문단위><조문번호>1</조문번호><조문가지번호>0</조문가지번호>
<조문내용>제1조(목적) 원자력 안전</조문내용></조문단위>
<조문단위><조문번호>53</조문번호><조문가지번호>3</조문가지번호>
<조문시행일자>20260519</조문시행일자>
<조문내용>제53조의3(방사선안전관리자) 안전 의무</조문내용></조문단위>
<조문단위><조문번호>99</조문번호><조문가지번호>0</조문가지번호>
<조문시행일자>20270220</조문시행일자>
<조문내용>미래 조문</조문내용></조문단위>
</조문></법령>""".encode("utf-8")


@pytest.fixture
def saved_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    write_official_law_corpus(
        root, XML,
        OfficialLawVersionRequest("011435", "286119", "2026-05-19", "2026-10-05"),
        required_articles=("53-3",),
    )
    return root


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None


class FakeTransaction:
    def __init__(self, existing):
        self.rows = dict(existing)

    def run(self, query, **params):
        if query == ARTICLE_DUPLICATE_QUERY:
            props = self.rows.get(params["article_key"])
            if props is None:
                return FakeCursor([])
            return FakeCursor([{
                "source_id": props["sourceChunkId"],
                "text": props["text"],
                "response_sha256": props["sourceResponseSha256"],
            }])
        assert query == CREATE_OFFICIAL_CHUNK_QUERY
        props = params["properties"]
        self.rows[props["articleKey"]] = props
        return FakeCursor([{"source_id": props["sourceChunkId"]}])


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute_write(self, func, chunks):
        tx = FakeTransaction(self.driver.rows)
        result = func(tx, chunks)
        # Emulate commit on success and rollback when a transaction raises.
        self.driver.rows = tx.rows
        return result


class FakeDriver:
    def __init__(self, rows=None):
        self.rows = rows or {}
        self.database = None

    def session(self, *, database):
        self.database = database
        return FakeSession(self)


def test_preflight_verifies_every_packet_and_excludes_future(saved_corpus):
    manifest, chunks = prepare_official_corpus_chunks(
        saved_corpus, required_articles=("53-3",)
    )
    assert manifest["article_count_eligible"] == 2
    assert manifest["article_count_excluded_future"] == 1
    assert len(chunks) == 2
    assert all(c["sourceStatus"] == "verified_official_xml" for c in chunks)
    assert all("art:99" not in c["articleKey"] for c in chunks)


def test_atomic_batch_create_and_idempotent_retry(saved_corpus):
    _, chunks = prepare_official_corpus_chunks(saved_corpus)
    driver = FakeDriver()
    first = import_verified_official_corpus(driver, chunks, database="derived")
    assert first == {
        "status": "applied_to_disposable_restore",
        "created": 2, "already_present": 0, "total": 2,
    }
    second = import_verified_official_corpus(driver, chunks, database="derived")
    assert second["created"] == 0
    assert second["already_present"] == 2
    assert len(driver.rows) == 2
    assert driver.database == "derived"


def test_batch_conflict_rolls_back_first_new_article(saved_corpus):
    _, chunks = prepare_official_corpus_chunks(saved_corpus)
    conflicting = {
        **chunks[1],
        "sourceChunkId": "old-unverified",
        "sourceResponseSha256": "a" * 64,
        "text": "conflicting",
    }
    driver = FakeDriver({chunks[1]["articleKey"]: conflicting})
    with pytest.raises(ValueError, match="Manual source reconciliation"):
        import_verified_official_corpus(driver, chunks, database="derived")
    assert len(driver.rows) == 1
    assert chunks[0]["articleKey"] not in driver.rows


def test_requires_verified_properties_and_unique_article_keys(saved_corpus):
    _, chunks = prepare_official_corpus_chunks(saved_corpus)
    with pytest.raises(ValueError, match="Duplicate"):
        import_verified_official_corpus(FakeDriver(), chunks + chunks[:1], database="x")
    with pytest.raises(ValueError, match="unverified"):
        import_verified_official_corpus(
            FakeDriver(), [{**chunks[0], "sourceStatus": "untrusted"}], database="x"
        )


def test_tampered_packet_rejected_before_neo4j_session(saved_corpus):
    path = saved_corpus / "articles/article-005303.json"
    packet = json.loads(path.read_text())
    packet["article_text"] = "tampered"
    path.write_text(json.dumps(packet, ensure_ascii=False))
    with pytest.raises(ValueError, match="differs"):
        prepare_official_corpus_chunks(saved_corpus)


def test_required_article_missing_or_future_rejected(saved_corpus):
    with pytest.raises(ValueError, match="Required"):
        prepare_official_corpus_chunks(saved_corpus, required_articles=("99",))


def test_report_never_pretends_overlay_is_a_snapshot(saved_corpus):
    manifest, _ = prepare_official_corpus_chunks(saved_corpus)
    report = build_official_corpus_overlay_report(
        manifest,
        {"base_snapshot_sha256": "a" * 64, "base_snapshot_stage": "community-ready"},
        result={"status": "applied_to_disposable_restore",
                "created": 2, "already_present": 0, "total": 2},
    )
    assert report["derived_snapshot_required"] is True
    assert report["graph_edges_and_communities_rebuilt"] is False
    assert report["official_source_xml_sha256"] == manifest["source_xml_sha256"]
