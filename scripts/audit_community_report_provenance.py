from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_drift.community_report_generation import (
    enrich_cached_community_report_provenance,
)
from graphrag_drift.community_report_store import FileCommunityReportStore
from graphrag_drift.community_report_temporal_provenance import (
    evaluate_community_source_date,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit cached report source law effective dates without an LLM."
    )
    parser.add_argument("--data-repo", default=".")
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Add missing input-source provenance to matching cached reports.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.data_repo)
    enrichment = None
    if args.enrich:
        enrichment = enrich_cached_community_report_provenance(root)

    dump_sha = json.loads(
        (root / "data/manifest.json").read_text(encoding="utf-8")
    )["dump"]["zip_sha256"]
    manifest = Path("community_reports") / dump_sha[:12] / "manifest.json"
    reports = FileCommunityReportStore.from_relative_manifest(root, manifest).load()

    results = []
    for report in reports:
        evaluation = evaluate_community_source_date(
            report.source_provenance, as_of_date=args.as_of,
        )
        results.append({
            "community_id": report.id,
            **evaluation,
        })
    payload = {
        "format": "graphrag-drift-community-provenance-audit-v1",
        "as_of_date": args.as_of,
        "report_count": len(results),
        "future_version_report_count": sum(
            row["status"] == "future_versions_in_report_input"
            for row in results
        ),
        "unverified_report_count": sum(
            row["status"] == "source_provenance_not_recorded"
            for row in results
        ),
        "enrichment": enrichment,
        "reports": results,
        "caveat": (
            "This audits the dated input context, not claim-level citations "
            "or legal correctness of precomputed reports."
        ),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
