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
  --output-json /tmp/kr-law-011435-53-3-20260519.json
```

The collector requests `target=eflaw&type=XML&MST=286119&efYd=20260519&JO=005303`,
validates the law ID, version effective date, exact article number and
article-specific effective date, and exports a JSON source packet with
an original-XML SHA-256. The `OC` value is never included in the
output JSON.

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
