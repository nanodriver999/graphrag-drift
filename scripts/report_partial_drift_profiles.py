from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_drift.benchmark_suite import load_benchmark_cases
from graphrag_drift.partial_profile_report import (
    build_partial_profile_report,
    render_partial_profile_report_markdown,
)
from graphrag_drift.profile_progress import ProfileBenchmarkProgress


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report completed paired DRIFT results without calling an LLM."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--progress-json", required=True)
    parser.add_argument("--profiles", nargs="+", default=["economy", "microsoft-like"])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    progress_path = Path(args.progress_json)
    if not progress_path.is_file():
        parser.error(f"No progress.json at {progress_path}")

    # Preserve strict identity in the original file; no write is performed.
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    identity = payload.get("identity") if isinstance(payload, dict) else None
    progress = ProfileBenchmarkProgress.load(progress_path, identity=identity)
    report = build_partial_profile_report(
        load_benchmark_cases(args.cases),
        progress,
        profiles=tuple(args.profiles),
    )
    rendered_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    rendered_md = render_partial_profile_report_markdown(report)

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered_json, encoding="utf-8")
    else:
        print(rendered_json)
    if args.output_markdown:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered_md, encoding="utf-8")
    else:
        print(rendered_md)


if __name__ == "__main__":
    main()
