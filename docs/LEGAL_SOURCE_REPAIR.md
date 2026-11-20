# Rebuild missing date-effective nuclear law article source

Strict benchmark #9 (2026-10-05 legal cutoff) confirmed that the restored
Neo4j snapshot lacked a **date-effective** version of
`원자력안전법` (law ID `011435`) and Article `53-3`.
The benchmark had to rely on its Enforcement Decree/Rules and correctly
reported insufficient primary-law source evidence.

The Korean National Law Information Center has the official
`원자력안전법` version **law no. 21679 effective 2026-05-19**,
with `lsiSeq=286119`, containing
`제53조의3(방사선안전관리자)`:
https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=286119

**Important:** the official web page displays some future-effective
amendment text (including a separate 2027 article version). Do **not**
ingest it as of 2026-10-05. Retrieve the exact effective-date version
with the government `eflaw` API and validate its article effective date.

## 1. Collect a verified source

The official government API requires an `OC` value issued to an API
user: https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsEfYdInfoGuide

```bash
# Provide your own credential as an environment secret. Do not print it.
export LAW_GO_KR_OC="<credential>"
python scripts/fetch_official_effective_law_article.py \
  --law-id 011435 \
  --mst 286119 \
  --article 53-3 \
  --effective-date 2026-05-19 \
  --as-of-date 2026-10-05 \
  --output-json /tmp/kr-law-011435-53-3-20260519.json \
  --output-xml /tmp/kr-law-011435-53-3-20260519.xml
```

The collector requests `target=eflaw&type=XML&MST=286119&efYd=20260519&JO=005303`,
validates the law ID, version effective date, exact article number and
article-specific effective date, and exports both the **original XML** and
a JSON source packet with its SHA-256. The `OC` value is never included
in either output. The separate XML must be retained for an independent
integrity check before any graph ingestion.

**The collector has no Neo4j write permissions or snapshot mutation.**
Its output contains `ingestion_status: downloaded_not_imported`.

## 2. Repair data in a derived source version

The current `community-ready` snapshot has a fixed SHA-256. Never
modify it in place or reuse the historical benchmark run ID after a
corpus change.

After the authenticated source download succeeds:

1. Check the official law version's full article text, paragraph/item
   numbering and amendment-specific effective dates.
2. Add the source as provenance-tracked `Chunk` records with
   `lawId=011435`, `articleKey` ending in `effective:20260519:art:53-3`,
   `officialEffectiveDate=20260519` and a verifiable official source URL.
   The entire 2026-effective law version may be needed to fix other
   benchmark coverage gaps; adding one clause is not the full correction.
3. Derive a **new** Neo4j snapshot with new manifest SHA-256, reconstruct
   graph relationships, communities, and Community Reports; audit
   Community Report source versions before reuse.
4. Run `scripts/check_aura_temporal_coverage.py --as-of 2026-10-05
   --law-id 011435 --article 011435:53-3 --require-current`
   against the newly restored snapshot.
5. Start new immutable, same-model DRIFT profile comparisons only after
   the new corpus passes coverage preflight. Do not mix old and new
   cached results.

The collection step is deliberately separate from ingestion to avoid
silently introducing unverified content or mislabelled future law into
the graph.


## 3. Stage an official `Chunk` overlay (dry run, no graph mutation)

A verified official article can now be mapped to the same versioned
Neo4j `Chunk` fields as the historical logical dump, using the *original*
XML bytes to independently validate every source JSON field.

```bash
python scripts/ingest_verified_official_law_article.py \
  --packet-json /tmp/kr-law-011435-53-3-20260519.json \
  --source-xml /tmp/kr-law-011435-53-3-20260519.xml \
  --base-snapshot-manifest data-repo/snapshots/community-ready/manifest.json \
  --base-snapshot-dump data-repo/snapshots/community-ready/neo4j.dump \
  --output-json /tmp/official-law-overlay-plan.json
```

This only produces a staging plan. It does **not** change the source dump,
the restored Neo4j database, or Community Reports.

If and only if a *separate, disposable* Neo4j instance has already been
restored from that original community-ready dump:

```bash
export LOCAL_NEO4J_URI=bolt://127.0.0.1:17687
export LOCAL_NEO4J_USERNAME=neo4j
export LOCAL_NEO4J_PASSWORD="<password>"
export LOCAL_NEO4J_DATABASE=neo4j

python scripts/ingest_verified_official_law_article.py \
  --packet-json /tmp/kr-law-011435-53-3-20260519.json \
  --source-xml /tmp/kr-law-011435-53-3-20260519.xml \
  --base-snapshot-manifest data-repo/snapshots/community-ready/manifest.json \
  --base-snapshot-dump data-repo/snapshots/community-ready/neo4j.dump \
  --output-json /tmp/derived-law-overlay-result.json \
  --apply --confirm-disposable-restore
```

The importer refuses remote Neo4j endpoints, requires a matching parent
dump SHA-256 and matching XML/JSON, and performs a single Neo4j write
transaction. Exact re-imports are idempotent; if any existing
`Chunk.articleKey` conflicts, the import fails instead of overwriting
evidence. The new `Chunk` is immediately eligible for local lexical
retrieval when the existing chunk full-text index is rebuilt.

**Source integrity limitation:** matching XML and JSON verifies internal
consistency and that the packet is the same one obtained by the
authenticated collector. It is not a digital signature by the legal
publisher. Protect the provenance files and record their digests.

**Data limitation:** adding only Article 53-3 into a derived Neo4j copy
does **not** repair the whole 2026-effective `원자력안전법`, rebuild the
community memberships, or make mixed-date Community Reports correct.
Do not use an article-only patched copy as a finalized production source.

## 4. Rebuild and validate a *new* snapshot

After imported text and legal scope are checked, rebuild graph indexes,
relationships/community clustering, and date-specific Community Reports
from the derived source; export a fresh `neo4j.dump` with a **new
manifest and SHA-256**. Never replace the historical
`snapshots/community-ready/neo4j.dump` in place.

Run on the **derived** database:

```bash
python scripts/check_aura_temporal_coverage.py \
  --as-of 2026-10-05 --law-id 011435 \
  --article 011435:53-3 --require-current
```

The first step of this workflow is implemented and covered by offline
tests. Full derived-dump rebuilding and new Community Report generation
must be completed separately before claiming a scientifically controlled
A/B search-policy result.
