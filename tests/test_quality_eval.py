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
