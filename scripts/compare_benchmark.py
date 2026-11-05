from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_drift.benchmark_regression import (
    RegressionThresholds,
    compare_benchmark_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("current")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-wall-increase-ratio", type=float, default=0.50)
    parser.add_argument("--max-llm-call-increase", type=float, default=1.0)
    parser.add_argument("--max-topic-coverage-drop", type=float, default=0.10)
    parser.add_argument("--max-insufficient-rate-increase", type=float, default=0.10)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))

    result = compare_benchmark_results(
        baseline,
        current,
        thresholds=RegressionThresholds(
            wall_ms_increase_ratio=args.max_wall_increase_ratio,
            llm_calls_increase=args.max_llm_call_increase,
            topic_coverage_drop=args.max_topic_coverage_drop,
            insufficient_rate_increase=args.max_insufficient_rate_increase,
        ),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    if args.fail_on_regression and result["status"] == "regressed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
