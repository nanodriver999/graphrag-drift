from __future__ import annotations

import hashlib
import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from graphrag_drift.official_law_api import (
    OfficialLawArticleRequest,
    fetch_official_article,
    parse_official_article_xml,
)


SOURCE_XML = """<?xml version="1.0" encoding="utf-8"?>
<법령>
  <기본정보>
    <법령ID>11435</법령ID>
    <법령명_한글>원자력안전법</법령명_한글>
    <시행일자>20260519</시행일자>
  </기본정보>
  <조문>
    <조문단위>
      <조문번호>53</조문번호>
      <조문가지번호>3</조문가지번호>
      <조문시행일자>20260519</조문시행일자>
      <조문제목>방사선안전관리자</조문제목>
      <조문내용>제53조의3(방사선안전관리자)</조문내용>
      <항><항내용>① 안전관리규정 준수 여부를 점검한다.</항내용></항>
      <항><항내용>② 안전관리조치를 권고한다.</항내용></항>
    </조문단위>
  </조문>
</법령>
""".encode("utf-8")


@pytest.fixture
def version() -> OfficialLawArticleRequest:
    return OfficialLawArticleRequest(
        law_id="011435",
        mst="286119",
        article="53-3",
        effective_date="2026-05-19",
        as_of_date="2026-10-05",
    )


def test_official_eflaw_url_encodes_exact_version_without_id_lookup(version) -> None:
    url = version.api_url(oc="example-credential")
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert parsed.hostname == "www.law.go.kr"
    assert params == {
        "OC": ["example-credential"],
        "target": ["eflaw"],
        "type": ["XML"],
        "MST": ["286119"],
        "efYd": ["20260519"],
        "JO": ["005303"],
    }
    assert "example-credential" not in version.public_source_url


def test_official_article_parser_preserves_version_and_source_hash(version) -> None:
    result = parse_official_article_xml(SOURCE_XML, version)
    assert result["law_id"] == "011435"
    assert result["law_name"] == "원자력안전법"
    assert result["article"] == "53-3"
    assert result["effective_date"] == "20260519"
    assert result["source"]["mst"] == "286119"
    assert result["source"]["jo"] == "005303"
    assert result["source"]["response_sha256"] == hashlib.sha256(SOURCE_XML).hexdigest()
    assert "안전관리규정 준수" in result["article_text"]
    assert "안전관리조치를 권고" in result["article_text"]
    assert result["ingestion_status"] == "downloaded_not_imported"
    assert "OC" not in json.dumps(result, ensure_ascii=False)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, limit: int) -> bytes:
        return self.body


def test_fetch_uses_authorized_url_but_never_returns_secret(version) -> None:
    with patch(
        "graphrag_drift.official_law_api.urlopen",
        return_value=FakeResponse(SOURCE_XML),
    ) as mocked:
        result = fetch_official_article(version, oc="private-key")
    req = mocked.call_args.args[0]
    assert "private-key" in req.full_url
    assert "private-key" not in json.dumps(result, ensure_ascii=False)
    assert mocked.call_args.kwargs["timeout"] == 30.0


def test_fetch_rejects_wrong_law_id_and_wrong_effective_version(version) -> None:
    with pytest.raises(ValueError, match="law ID"):
        parse_official_article_xml(
            SOURCE_XML.replace(b"11435", b"11436"), version,
        )
    with pytest.raises(ValueError, match="effective date"):
        parse_official_article_xml(
            SOURCE_XML.replace(b"20260519", b"20270220"), version,
        )


def test_fetch_rejects_future_article_even_in_2026_law(version) -> None:
    bad = SOURCE_XML.replace(
        b"<조문시행일자>20260519</조문시행일자>",
        b"<조문시행일자>20270101</조문시행일자>",
    )
    with pytest.raises(ValueError, match="future"):
        parse_official_article_xml(bad, version)


def test_fetch_rejects_missing_target_article(version) -> None:
    bad = SOURCE_XML.replace(
        b"<조문가지번호>3</조문가지번호>",
        b"<조문가지번호>2</조문가지번호>",
    )
    with pytest.raises(ValueError, match="Expected one article"):
        parse_official_article_xml(bad, version)


def test_future_law_version_cannot_be_requested_for_earlier_cutoff() -> None:
    with pytest.raises(ValueError, match="after as-of"):
        OfficialLawArticleRequest(
            law_id="011435", mst="283347", article="53-3",
            effective_date="2027-02-20", as_of_date="2026-10-05",
        )


def test_api_authentication_is_required(version) -> None:
    with pytest.raises(ValueError, match="authentication"):
        version.api_url(oc="")
