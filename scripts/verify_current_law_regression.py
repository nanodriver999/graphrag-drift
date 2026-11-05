from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graphrag_drift.current_law_regression import verify_current_law_regression


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Conditionally verify Local Search once the current-effective law "
            "version is present in Aura."
        )
    )
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--local-result", required=True)
    parser.add_argument("--law-id", required=True)
    parser.add_argument("--article-fragment", required=True)
    args = parser.parse_args()

    result = verify_current_law_regression(
        _load(args.coverage),
        _load(args.local_result),
        law_id=args.law_id,
        required_article_fragment=args.article_fragment,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
