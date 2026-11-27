from __future__ import annotations

import argparse
import json
import os

from graphrag_drift.official_law_corpus import (
    OfficialLawVersionRequest,
    fetch_official_law_xml,
    verify_official_law_corpus,
    write_official_law_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect an exact date-effective whole Korean law version, "
            "preserve original XML, and validate every included article. "
            "Does not change Neo4j or historical snapshots."
        )
    )
    parser.add_argument("--verify-dir", help="Verify existing corpus offline, without an API key")
    parser.add_argument("--law-id", help="Exact official numeric law ID")
    parser.add_argument("--mst", help="Exact official law version MST / lsiSeq")
    parser.add_argument("--effective-date", help="YYYY-MM-DD for the requested law version")
    parser.add_argument("--as-of-date", help="YYYY-MM-DD cutoff used for the benchmark")
    parser.add_argument("--output-dir", help="New empty destination for immutable XML and packets")
    parser.add_argument("--require-article", action="append", default=[],
                        help="Article that must be present and date-effective, e.g. 53-3")
    args = parser.parse_args()
    if args.verify_dir:
        if any([args.law_id, args.mst, args.effective_date, args.as_of_date,
                args.output_dir, args.require_article]):
            parser.error("--verify-dir cannot be combined with fetch arguments")
        manifest = verify_official_law_corpus(args.verify_dir)
    else:
        if not all([args.law_id, args.mst, args.effective_date,
                    args.as_of_date, args.output_dir]):
            parser.error("law-id, mst, effective-date, as-of-date and output-dir are required")
        oc = os.environ.get("LAW_GO_KR_OC")
        if not oc:
            parser.error("LAW_GO_KR_OC secret must be set for authenticated official collection")
        request = OfficialLawVersionRequest(
            law_id=args.law_id,
            mst=args.mst,
            effective_date=args.effective_date,
            as_of_date=args.as_of_date,
        )
        raw = fetch_official_law_xml(request, oc=oc)
        manifest = write_official_law_corpus(
            args.output_dir, raw, request, required_articles=tuple(args.require_article)
        )
    print(json.dumps({
        "law_id": manifest["law_id"],
        "mst": manifest["mst"],
        "as_of_date": manifest["as_of_date"],
        "source_xml_sha256": manifest["source_xml_sha256"],
        "article_count_total": manifest["article_count_total"],
        "article_count_eligible": manifest["article_count_eligible"],
        "article_count_excluded_future": manifest["article_count_excluded_future"],
        "status": "verified_official_corpus_not_imported",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
