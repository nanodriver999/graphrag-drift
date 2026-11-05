from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INSUFFICIENT_MARKERS = (
    "증거만으로는",
    "근거가 부족",
    "확인할 수 없습니다",
    "insufficient evidence",
    "no relevant local evidence",
)


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _has_current_law(coverage: dict[str, Any], law_id: str) -> bool:
    for row in coverage.get("results", []):
        if str(row.get("law_id")) == law_id:
            return bool(row.get("has_current_version"))
    return False


def _evidence_ids(local_result: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in local_result.get("evidence", []):
        if isinstance(item, dict):
            value = item.get("id")
            if isinstance(value, str):
                ids.append(value)
    return ids


def verify_current_law_regression(
    coverage: dict[str, Any],
    local_result: dict[str, Any],
    *,
    law_id: str,
    required_article_fragment: str,
) -> dict[str, Any]:
    current_present = _has_current_law(coverage, law_id)
    evidence_ids = _evidence_ids(local_result)
    answer = str(local_result.get("answer") or "").strip()

    if not current_present:
        return {
            "status": "skipped",
            "law_id": law_id,
            "reason": "current-effective version is not present",
            "evidence_ids": evidence_ids,
        }

    if not answer:
        raise AssertionError("Local Search returned an empty answer")

    if not any(required_article_fragment in item_id for item_id in evidence_ids):
        raise AssertionError(
            f"Current law {law_id} is present but required article "
            f"{required_article_fragment!r} was not retrieved: {evidence_ids}"
        )

    lowered = answer.lower()
    if any(marker.lower() in lowered for marker in INSUFFICIENT_MARKERS):
        raise AssertionError(
            "Current governing article was retrieved but the answer still reports "
            f"insufficient evidence: {answer}"
        )

    return {
        "status": "passed",
        "law_id": law_id,
        "required_article_fragment": required_article_fragment,
        "evidence_ids": evidence_ids,
        "answer": answer,
    }


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
