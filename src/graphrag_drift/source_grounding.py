from __future__ import annotations

import re
from typing import Any, Iterable

from .temporal_coverage import normalize_date


# The retriever includes the law version and article in Chunk.articleKey.
# Do not accept Entity or LLM-created IDs as direct official-article evidence.
_SOURCE_ARTICLE_ID = re.compile(
    r"^chunk-article:kr:law:(?P<law_id>[0-9]+):"
    r".*:effective:(?P<effective_date>[0-9]{8}):"
    r"art:(?P<article>[0-9]+(?:-[0-9]+)?)$"
)
_EXPECTED_ARTICLE = re.compile(r"^(?P<law_id>[0-9]+):(?P<article>[0-9]+(?:-[0-9]+)?)$")


def parse_article_source_id(source_id: str) -> dict[str, str] | None:
    """Parse explicit versioned article source identity from a retrieved hit."""
    match = _SOURCE_ARTICLE_ID.fullmatch(source_id)
    if not match:
        return None
    return {
        "law_id": match["law_id"],
        "article": match["article"],
        "effective_date": match["effective_date"],
        "source_id": source_id,
    }


def collect_article_sources(source_ids: Iterable[str]) -> list[dict[str, str]]:
    """Deduplicate physical article references, retaining effective dates."""
    found: dict[str, dict[str, str]] = {}
    for source_id in source_ids:
        source = parse_article_source_id(source_id)
        if source is not None:
            found[source_id] = source
    return [found[key] for key in sorted(found)]


def _answer_mentions_article(answer: str, article: str) -> bool:
    # 53-3 is article 53-3 (제53조의3), not clause 3 of article 53.
    number, *sub = article.split("-")
    expression = rf"제\s*{re.escape(number)}\s*조"
    if sub:
        expression += rf"\s*의\s*{re.escape(sub[0])}(?![0-9])"
    else:
        expression += r"(?!\s*의\s*[0-9])"
    return re.search(expression, answer) is not None


def evaluate_article_source_support(
    answer: str,
    source_articles: list[dict[str, str]],
    expected_articles: list[str],
    *,
    as_of_date: str,
) -> dict[str, Any]:
    """Check direct-source retrieval coverage; not semantic entailment.

    Citation-to-source matching here only verifies that a correctly dated
    article *was retrieved*. It cannot prove that the article supports the
    answer's factual assertions.
    """
    as_of = normalize_date(as_of_date)
    expected = []
    for item in expected_articles:
        match = _EXPECTED_ARTICLE.fullmatch(item)
        if match is None:
            raise ValueError(
                f"Expected article must use law-id:article-number, got {item!r}"
            )
        expected.append((item, match["law_id"], match["article"]))

    matched: list[str] = []
    future_only: list[str] = []
    undated: list[str] = []
    ungrounded_mentions: list[str] = []
    for label, law_id, article in expected:
        relevant = [
            entry for entry in source_articles
            if entry.get("law_id") == law_id and entry.get("article") == article
        ]
        dated = [
            entry for entry in relevant
            if re.fullmatch(r"[0-9]{8}", entry.get("effective_date", ""))
        ]
        current = [
            entry for entry in dated
            if entry["effective_date"] <= as_of
        ]
        if current:
            matched.append(label)
        elif any(entry["effective_date"] > as_of for entry in dated):
            future_only.append(label)
        elif relevant:
            undated.append(label)
        if not current and _answer_mentions_article(answer, article):
            ungrounded_mentions.append(label)

    missing = [label for label, _, _ in expected if label not in matched]
    return {
        "expected_article_count": len(expected),
        "direct_article_source_coverage": (
            round(len(matched) / len(expected), 3) if expected else None
        ),
        "matched_expected_article_sources": matched,
        "missing_expected_article_sources": missing,
        "future_only_expected_article_sources": future_only,
        "undated_expected_article_sources": undated,
        # A retrieval warning; not a hallucination judgement.
        "expected_article_mentions_without_direct_source": ungrounded_mentions,
        "source_as_of_date": as_of,
    }
