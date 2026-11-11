from __future__ import annotations

import pytest

from graphrag_drift.benchmark_suite import BenchmarkCase
from scripts.run_drift_profile_comparison import select_benchmark_cases


def test_select_benchmark_cases_retains_dataset_order() -> None:
    cases = [
        BenchmarkCase("first", "question one"),
        BenchmarkCase("second", "question two"),
        BenchmarkCase("third", "question three"),
    ]

    selected = select_benchmark_cases(cases, ["third", "first", "third"])

    assert [case.id for case in selected] == ["first", "third"]
    assert select_benchmark_cases(cases, []) is cases


def test_select_benchmark_cases_rejects_unknown_ids() -> None:
    cases = [BenchmarkCase("first", "question one")]

    with pytest.raises(ValueError, match="Unknown benchmark case ID\\(s\\): missing"):
        select_benchmark_cases(cases, ["missing"])
