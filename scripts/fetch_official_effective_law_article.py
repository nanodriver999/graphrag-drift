from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from graphrag_drift.official_law_api import (
    OfficialLawArticleRequest,
    fetch_official_article,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download an exact date-effective official law article for source "
            "ingestion. Does not modify Neo4j or an existing snapshot."
        ),
    )
    parser.add_argument("--law-id", required=True)
    parser.add_argument("--mst", required=True)
    parser.add_argument("--article", required=True)
    parser.add_argument("--effective-date", required=True)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    oc = os.getenv("LAW_GO_KR_OC")
    if not oc:
        parser.error("LAW_GO_KR_OC must be set to a valid law.go.kr OC credential")
    request = OfficialLawArticleRequest(
        law_id=args.law_id,
        mst=args.mst,
        article=args.article,
        effective_date=args.effective_date,
        as_of_date=args.as_of_date,
    )
    result = fetch_official_article(request, oc=oc)
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Verified official {result['law_name']} article {result['article']}, "
        f"version effective {result['effective_date']}; wrote {path}. "
        "Source is NOT yet imported into any Neo4j snapshot."
    )


if __name__ == "__main__":
    main()
