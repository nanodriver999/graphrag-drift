import pytest

from graphrag_drift.quality_eval import evaluate_answer_quality


def test_quality_eval_ignores_whitespace_and_case() -> None:
    result = evaluate_answer_quality(
        "Safety Manager 선임 신고 절차",
        ["safety manager", "선임신고", "교육"],
    )

    assert result["matched_topics"] == ["safety manager", "선임신고"]
    assert result["topic_coverage"] == 0.667
    assert not result["insufficient_answer"]


def test_quality_eval_detects_insufficient_answer() -> None:
    result = evaluate_answer_quality(
        "제공된 증거만으로는 주요 의무를 확인할 수 없습니다.",
        ["주요 의무"],
    )

    assert result["insufficient_answer"]
    assert result["topic_coverage"] == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        "제공된 증거에는 해당 의무를 직접 규정한 조항이 없어 특정할 수 없습니다.",
        "증거에는 구체적인 직무 의무가 명시되어 있지 않습니다.",
        "업무대행자에게 적용되는지는 근거상 불명확합니다.",
    ],
)
def test_quality_eval_recognizes_real_benchmark_insufficient_phrasings(
    answer: str,
) -> None:
    result = evaluate_answer_quality(answer, ["안전관리규정", "조치"])
    assert result["insufficient_answer"]


def test_quality_eval_does_not_flag_positive_answer() -> None:
    result = evaluate_answer_quality(
        "방사선안전관리자에게 교육 이수 및 선임신고 의무가 있습니다.",
        ["교육", "선임신고"],
    )
    assert not result["insufficient_answer"]
