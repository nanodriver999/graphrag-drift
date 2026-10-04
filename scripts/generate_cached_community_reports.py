from __future__ import annotations

import argparse
import json
import os

from graphrag_drift.community_report_generation import CommunityReportCacheBuilder
from graphrag_drift.openai_compatible import OpenAICompatibleTextGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or reuse versioned Community Report cache files."
    )
    parser.add_argument("--data-repo", default=".")
    parser.add_argument(
        "--generator-name",
        default=os.getenv(
            "COMMUNITY_REPORT_GENERATOR_NAME",
            "opencode-muse-spark-1.3-contributor-free",
        ),
    )
    args = parser.parse_args()

    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
    model = os.environ.get("OPENAI_COMPATIBLE_MODEL")
    if not base_url or not model:
        raise RuntimeError(
            "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL are required"
        )

    generator = OpenAICompatibleTextGenerator(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("OPENAI_COMPATIBLE_API_KEY"),
        timeout_seconds=float(os.getenv("OPENAI_COMPATIBLE_TIMEOUT_SECONDS", "180")),
        temperature=float(os.getenv("OPENAI_COMPATIBLE_TEMPERATURE", "0")),
    )
    result = CommunityReportCacheBuilder(
        generator=generator,
        generator_name=args.generator_name,
    ).build(args.data_repo)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
