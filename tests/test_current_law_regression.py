from __future__ import annotations

import pytest

from graphrag_drift.current_law_regression import verify_current_law_regression


def coverage(current: bool) -> dict:
    return {
        "results": [
            {
                "law_id": "011435",
                "has_current_version": current,
            }
        ]
    }


def local(ids: list[str], answer: str) -> dict:
    return {
        "evidence": [{"id": item_id, "text": "x", "score": 1.0} for item_id in ids],
        "answer": answer,
    }


def test_skips_when_current_version_is_missing() -> None:
    result = verify_current_law_regression(
        coverage(False),
        local(["chunk-article:other"], "근거 부족"),
        law_id="011435",
        required_article_fragment=":art:53-3",
    )

    assert result["status"] == "skipped"


def test_passes_when_current_article_is_retrieved_and_answer_is_grounded() -> None:
    result = verify_current_law_regression(
        coverage(True),
        local(
            ["chunk-article:kr:law:011435:mst:x:effective:20260519:art:53-3"],
            "방사선안전관리자는 안전관리규정 준수 여부를 점검한다.",
        ),
        law_id="011435",
        required_article_fragment=":art:53-3",
    )

    assert result["status"] == "passed"


def test_fails_when_current_version_exists_but_article_is_not_retrieved() -> None:
    with pytest.raises(AssertionError, match="was not retrieved"):
        verify_current_law_regression(
            coverage(True),
            local(["chunk-article:other"], "답변"),
            law_id="011435",
            required_article_fragment=":art:53-3",
        )


def test_fails_when_answer_still_reports_insufficient_evidence() -> None:
    with pytest.raises(AssertionError, match="insufficient evidence"):
        verify_current_law_regression(
            coverage(True),
            local(
                ["chunk-article:kr:law:011435:mst:x:effective:20260519:art:53-3"],
                "제공된 증거만으로는 확인할 수 없습니다.",
            ),
            law_id="011435",
            required_article_fragment=":art:53-3",
        )
