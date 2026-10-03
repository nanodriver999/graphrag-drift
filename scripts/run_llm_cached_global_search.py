#!/usr/bin/env python3
"""Run LLM-backed GraphRAG Global Search over cached Community Reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_drift import (
    GlobalSearchLLMSettings,
    build_llm_cached_global_search_engine,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument(
        "--data-repo",
        required=True,
        type=Path,
        help="Path to a checkout of nanodriver999/graphrag-drift-data",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = GlobalSearchLLMSettings.from_env()
    engine = build_llm_cached_global_search_engine(args.data_repo, settings)
    result = engine.global_search(args.query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
