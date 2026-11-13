from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .temporal_coverage import normalize_date

_API_URL = "https://www.law.go.kr/DRF/lawService.do"
_OFFICIAL_PAGE = "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq="


@dataclass(frozen=True)
class OfficialLawArticleRequest:
    law_id: str
    mst: str
    article: str
    effective_date: str
    as_of_date: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{1,8}", self.law_id):
            raise ValueError("law_id must be a numeric law ID")
        if not re.fullmatch(r"\d{1,10}", self.mst):
            raise ValueError("mst must be a numeric law version identifier")
        if not re.fullmatch(r"\d{1,4}(?:-\d{1,2})?", self.article):
            raise ValueError("article must be like 53-3")
        if normalize_date(self.effective_date) > normalize_date(self.as_of_date):
            raise ValueError("requested law version takes effect after as-of date")

    @property
    def jo(self) -> str:
        article, _, sub = self.article.partition("-")
        return f"{int(article):04d}{int(sub or 0):02d}"

    @property
    def public_source_url(self) -> str:
        return _OFFICIAL_PAGE + self.mst

    def api_url(self, *, oc: str) -> str:
        if not oc or not oc.strip():
            raise ValueError("LAW_GO_KR_OC authentication is required")
        return _API_URL + "?" + urlencode({
            "OC": oc.strip(),
            "target": "eflaw",
            "type": "XML",
            "MST": self.mst,
            "efYd": normalize_date(self.effective_date),
            "JO": self.jo,
        })


def _find_text(root: ET.Element, tag: str) -> str:
    node = root.find(".//" + tag)
    return "".join(node.itertext()).strip() if node is not None else ""


def _article_text(article_node: ET.Element) -> str:
    lines = []
    for node in article_node.iter():
        if node.tag in {"조문내용", "항내용", "호내용", "목내용"}:
            value = "".join(node.itertext()).strip()
            if value and value not in lines:
                lines.append(value)
    return "\n".join(lines)


def parse_official_article_xml(
    raw: bytes, request: OfficialLawArticleRequest,
) -> dict:
    """Fail closed on wrong/ambiguous law versions and future article clauses."""
    root = ET.fromstring(raw)
    raw_law_id = _find_text(root, "법령ID")
    effective = _find_text(root, "시행일자")
    law_name = _find_text(root, "법령명_한글") or _find_text(root, "법령명한글")
    if not raw_law_id or not raw_law_id.isdigit() or (
        int(raw_law_id) != int(request.law_id)
    ):
        raise ValueError("Official law ID is absent or does not match request")
    if not law_name:
        raise ValueError("Official law name is missing")
    if effective != normalize_date(request.effective_date):
        raise ValueError("Official returned law effective date differs from requested version")

    article, _, sub = request.article.partition("-")
    matches = [
        item for item in root.iter("조문단위")
        if _find_text(item, "조문번호") == str(int(article))
        and int(_find_text(item, "조문가지번호") or "0") == int(sub or 0)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one article {request.article} in versioned official response; "
            f"found {len(matches)}"
        )
    node = matches[0]
    article_effective = _find_text(node, "조문시행일자")
    if article_effective and (
        not re.fullmatch(r"\d{8}", article_effective)
        or article_effective > normalize_date(request.as_of_date)
    ):
        raise ValueError("Article has a future or invalid effective date")
    text = _article_text(node)
    if not text:
        raise ValueError("Official law article response contained no text")
    return {
        "format": "graphrag-drift-official-law-article-v1",
        "source": {
            "provider": "Korean National Law Information Center",
            "public_url": request.public_source_url,
            "api_target": "eflaw",
            "mst": request.mst,
            "jo": request.jo,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "law_id": request.law_id,
        "law_name": law_name,
        "article": request.article,
        "effective_date": effective,
        "article_effective_date": article_effective or effective,
        "as_of_date": normalize_date(request.as_of_date),
        "article_text": text,
        "ingestion_status": "downloaded_not_imported",
    }


def fetch_official_article_with_xml(
    request_info: OfficialLawArticleRequest,
    *,
    oc: str,
    timeout_seconds: float = 30.0,
) -> tuple[dict, bytes]:
    """Return the validated packet AND its original bytes for later verification."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    url = request_info.api_url(oc=oc)
    # Never log url: the authenticated query includes an OC secret.
    req = Request(url, headers={"User-Agent": "graphrag-drift-law-import/1.0"})
    with urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Official law response unexpectedly exceeds 4 MiB")
    return parse_official_article_xml(raw, request_info), raw


def fetch_official_article(
    request_info: OfficialLawArticleRequest,
    *,
    oc: str,
    timeout_seconds: float = 30.0,
) -> dict:
    packet, _raw = fetch_official_article_with_xml(
        request_info, oc=oc, timeout_seconds=timeout_seconds
    )
    return packet
