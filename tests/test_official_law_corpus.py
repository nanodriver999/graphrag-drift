from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

from graphrag_drift.official_law_api import (
    OfficialLawArticleRequest,
    parse_official_article_xml,
)
from graphrag_drift.official_law_corpus import (
    OfficialLawVersionRequest,
    fetch_official_law_xml,
    parse_official_law_corpus,
    verify_official_law_corpus,
    write_official_law_corpus,
)

XML = """<?xml version="1.0" encoding="utf-8"?>
<법령><기본정보>
<법령ID>11435</법령ID><법령명_한글>원자력안전법</법령명_한글>
<시행일자>20260519</시행일자>
<법령일련번호>286119</법령일련번호>
</기본정보>
<조문>
<조문단위><조문번호>1</조문번호><조문가지번호>0</조문가지번호>
<조문시행일자>20260519</조문시행일자>
<조문내용>제1조(목적)</조문내용><항><항내용>① 목적 설명</항내용></항>
</조문단위>
<조문단위><조문번호>53</조문번호><조문가지번호>3</조문가지번호>
<조문시행일자>20260519</조문시행일자>
<조문내용>제53조의3(방사선안전관리자)</조문내용>
<항><항내용>① 안전관리규정 확인</항내용></항>
</조문단위>
<조문단위><조문번호>70</조문번호><조문가지번호>0</조문가지번호>
<조문시행일자>20270220</조문시행일자>
<조문내용>제70조(미래 시행 조문)</조문내용>
</조문단위>
</조문></법령>
""".encode("utf-8")


@pytest.fixture
def version() -> OfficialLawVersionRequest:
    return OfficialLawVersionRequest(
        law_id="011435", mst="286119",
        effective_date="2026-05-19", as_of_date="2026-10-05"
    )


def test_whole_law_api_omits_jo_and_is_pinned_to_exact_version(version):
    args = parse_qs(urlsplit(version.api_url(oc="dummy-secret")).query)
    assert args == {
        "OC": ["dummy-secret"], "target": ["eflaw"], "type": ["XML"],
        "MST": ["286119"], "efYd": ["20260519"],
    }
    assert "JO" not in args


def test_parses_all_effective_articles_but_quarantines_future(version):
    corpus = parse_official_law_corpus(XML, version)
    assert set(corpus["packets"]) == {"1", "53-3"}
    assert corpus["article_count_total"] == 3
    assert corpus["excluded_future_articles"] == [{
        "article": "70", "effective_date": "20270220", "reason": "future_effective"
    }]
    packet = corpus["packets"]["53-3"]
    assert packet == parse_official_article_xml(
        XML,
        OfficialLawArticleRequest(
            "011435", "286119", "53-3", "2026-05-19", "2026-10-05"
        )
    )
    assert packet["ingestion_status"] == "downloaded_not_imported"
    assert "dummy-secret" not in json.dumps(packet)


def test_write_and_independently_verify_original_xml(tmp_path: Path, version):
    root = tmp_path / "official-corpus"
    manifest = write_official_law_corpus(root, XML, version, required_articles=("53-3",))
    assert manifest["article_count_eligible"] == 2
    assert manifest["article_count_excluded_future"] == 1
    assert (root / "source.xml").read_bytes() == XML
    assert (root / "articles/article-005303.json").is_file()
    assert verify_official_law_corpus(root) == manifest
    with pytest.raises(FileExistsError):
        write_official_law_corpus(root, XML, version)


def test_fails_if_required_article_absent_or_future(tmp_path: Path, version):
    with pytest.raises(ValueError, match="Required date-effective"):
        write_official_law_corpus(
            tmp_path / "missing", XML, version, required_articles=("70", "53-3")
        )
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("tamper", ["packet", "manifest", "xml", "extra"])
def test_modified_artifacts_are_rejected(tmp_path: Path, version, tamper):
    root = tmp_path / "official-corpus"
    write_official_law_corpus(root, XML, version)
    if tamper == "packet":
        path = root / "articles/article-005303.json"
        packet = json.loads(path.read_text())
        packet["article_text"] = "조작된 법령 내용"
        path.write_text(json.dumps(packet, ensure_ascii=False))
    elif tamper == "manifest":
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["article_count_total"] = 2
        path.write_text(json.dumps(manifest, ensure_ascii=False))
    elif tamper == "xml":
        (root / "source.xml").write_bytes(XML.replace(b"11435", b"11436"))
    else:
        (root / "articles" / "injected.json").write_text("{}")
    with pytest.raises(ValueError):
        verify_official_law_corpus(root)


def test_rejects_duplicate_and_invalid_article_number(version):
    duplicate = XML.decode("utf-8").replace(
        "<조문번호>70</조문번호>", "<조문번호>53</조문번호>"
    ).replace(
        "<조문가지번호>0</조문가지번호>\n<조문시행일자>20270220",
        "<조문가지번호>3</조문가지번호>\n<조문시행일자>20270220",
    ).encode("utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_official_law_corpus(duplicate, version)
    bad = XML.decode("utf-8").replace(
        "<조문번호>53</조문번호>", "<조문번호>XX</조문번호>"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="invalid"):
        parse_official_law_corpus(bad, version)


def test_wrong_law_or_wrong_effective_date_rejected(version):
    with pytest.raises(ValueError, match="law ID"):
        parse_official_law_corpus(XML.replace(b"11435", b"11436"), version)
    with pytest.raises(ValueError, match="effective date"):
        parse_official_law_corpus(XML.replace(b"20260519", b"20260919"), version)


def test_rejects_future_requested_version():
    with pytest.raises(ValueError, match="after as-of"):
        OfficialLawVersionRequest(
            "011435", "283347", "2027-02-20", "2026-10-05"
        )


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        return XML


def test_fetch_uses_official_api_without_leaking_secret(version):
    with patch(
        "graphrag_drift.official_law_corpus.urlopen", return_value=FakeResponse()
    ) as mocked:
        assert fetch_official_law_xml(version, oc="my-oc-secret") == XML
    assert "my-oc-secret" in mocked.call_args.args[0].full_url


def test_missing_articles_or_oversized_xml_fail_closed(version):
    with pytest.raises(ValueError, match="no article"):
        parse_official_law_corpus(XML.replace("조문단위".encode("utf-8"), "조문없음".encode("utf-8")), version)
    with pytest.raises(ValueError, match="exceeds"):
        parse_official_law_corpus(b"x" * (4 * 1024 * 1024 + 1), version)
