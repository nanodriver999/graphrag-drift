from __future__ import annotations

import re
from typing import Any


INSUFFICIENT_MARKERS = (
    "증거만으로는",
    "근거가 부족",
    "확인할 수 없습니다",
    "insufficient evidence",
    "no relevant",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def evaluate_answer_quality(
    answer: str,
    expected_topics: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    normalized = _normalize(answer)
    topics = [topic.strip() for topic in expected_topics if topic.strip()]
    matched = [topic for topic in topics if _normalize(topic) in normalized]
    coverage = 1.0 if not topics else len(matched) / len(topics)
    insufficient = any(
        _normalize(marker) in normalized
        for marker in INSUFFICIENT_MARKERS
    )
    return {
        "expected_topic_count": len(topics),
        "matched_topics": matched,
        "topic_coverage": round(coverage, 3),
        "insufficient_answer": insufficient,
    }
