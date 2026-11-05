from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_drift.benchmark_report import render_benchmark_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    markdown = render_benchmark_markdown(result)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
