# Community Report temporal source provenance

Community Reports contain LLM-synthesized summaries, so filtering only dated
`Chunk` hits does not establish the temporal validity of DRIFT primers.

The report-cache builder now derives `source_provenance` from the
`community_report_inputs/<community>/evidence_parts/` source JSON, preserving
the original `articleKey`, `lawId`, and `officialEffectiveDate` without
involving a language model. Existing cached summaries with matching context SHA
are upgraded in place without regenerating their text.

## Offline audit

From a checkout of the GraphRAG application and data repo:

```bash
python app/scripts/audit_community_report_provenance.py \
  --data-repo . \
  --as-of 2026-10-05 \
  --enrich \
  --output community-report-temporal-audit.json
```

The audit reports (per community):

- `future_article_keys`: source law articles effective **after** the benchmark cutoff
- `dated_article_count`: number of distinct versioned source articles
- `uncertain_source_count`: unversioned Chunks, non-Chunk evidence, and Entity descriptions
- `source_provenance_not_recorded` for legacy cache entries that have not been enriched

**Important:** a report whose input contains a future-effective article cannot
be considered date-safe for an earlier query. A report without such an article
also **cannot be certified safe**: Entity descriptions, undated documents and
synthesized claims may still contain temporally incorrect normative statements.

The cached report's LLM `content_hash` is unchanged during enrichment because
the input context did not change. The report generation code validates the same
context SHA before reusing report text. This is *source-input attribution*, not
claim-level proof or automatic redaction of mixed-date report paragraphs.

Next step: generate source-date-specific Community Reports from a filtered
snapshot and store/restore them with source provenance in Neo4j. Until that
happens, use this audit as a warning rather than treating every undated report
as reliable current law.


## Optional cached-report retrieval policies

`CachedCommunityReportRetriever` supports date-aware report filtering when
`as_of_date` is supplied:

- `temporal_policy="none"` (default): historical behavior; all cached reports rank
  normally, including reports with unknown temporal provenance.
- `temporal_policy="exclude_future"`: omit a report when its *input* contains
  any versioned law article effective after the cutoff. Legacy input-unknown
  reports remain retrievable and must not be treated as certified current law.
- `temporal_policy="require_date_clear_input"`: also omit input-unknown reports
  and reports with non-dated evidence or Entity descriptions. At least one
  dated original law article must exist in its input. This still **does not**
  establish claim-level entailment of the generated report text.

Example:

```python
retriever = CachedCommunityReportRetriever(
    local_retriever=neo4j_retriever,
    report_store=FileCommunityReportStore.from_relative_manifest(
        data_root, "community_reports/38228132813f/manifest.json"
    ),
    as_of_date="2026-10-05",
    temporal_policy="exclude_future",
)
```

These options affect the **file-cached Community Report adapter only**. The
default Neo4j-backed `COMMUNITY_REPORT_QUERY` still needs report-level source
metadata to support equivalent query-time filtering. An existing snapshot
cannot be retroactively declared legally current merely because a separate
cache was audited.


## Optional Neo4j Community Report allowlist

The Neo4j-backed report query now accepts a configurable, explicit
`allowed_community_ids` list. If the caller sets the environment variable
`GRAPHRAG_ALLOWED_COMMUNITY_IDS` when running
`scripts/run_drift_profile_comparison.py`, that list is passed to Neo4j:

```bash
# Generated from a prior source-date audit. Do NOT treat these IDs as
# legally verified just because their evidence had no future dates.
export GRAPHRAG_ALLOWED_COMMUNITY_IDS="community-search,leiden-1"
```

The configured IDs and corresponding selection are included in the strict
benchmark identity and LangGraph checkpoint fingerprint. **An unset variable**
retains the old unrestricted Community Report query. An **empty string** sets
an explicit empty list, preventing any Community Report from being returned;
DRIFT can fall back to local retrieval where supported.

Filtering by the IDs in a *cached input-source audit* does not prove those
same report IDs in a restored Neo4j snapshot have identical summaries or
source provenance. Match snapshot, dump and community IDs before interpreting
results, and eventually persist report source dates in Neo4j itself. This
allowlist enables controlled, explicitly limited experiments while that
metadata migration is pending.


## Avoiding irrelevant priming after temporal filtering

If the 2026-10-05 source audit excludes all legal communities,
the only remaining eligible reports may discuss GraphRAG itself.
Unrelated report summaries should not be used as legal DRIFT primers
solely because they have a nonzero global rank.

The Neo4j report retriever supports an **opt-in** minimum count of
normalized query terms present in the report summary or community title:

```bash
export GRAPHRAG_MIN_COMMUNITY_TERM_OVERLAP=1
```

A threshold of `1` rejects reports with no lexical overlap. If no
Community Report remains, the DRIFT workflow uses its existing local
search fallback; it must not invent report-derived legal evidence.

The threshold is recorded in the immutable comparison identity and
LangGraph fingerprint; changing it requires a separate run ID.
The default is `0` for backward compatibility with previously
published benchmark conditions.

**Limitation:** lexical matching is not semantic retrieval. Tokenization
may miss paraphrases or Korean compounds; this option is a conservative
noise gate, not a relevance/correctness guarantee. Prefer evaluating
relevance and source-coverage changes separately.


## Optional strict Entity description text gate

The Neo4j Hybrid/Vector Entity retrieval paths can now suppress the free-text
Entity.description when its legal source version cannot be verified. This
addresses a separate risk from filtering future-dated Chunk nodes: an old
Entity description can still summarize future law even if all attached Chunk
references are date-filtered.

For pinned strict DRIFT benchmarking, explicitly set:

    export GRAPHRAG_STRIP_UNVERIFIED_ENTITY_DESCRIPTIONS=true

The setting is included in the progress/cache benchmark identity so a run
with unverified descriptions can never be resumed silently as a strict run.
Default false preserves historical exploratory behavior.

When enabled, query text retains Entity names, graph relationship labels,
and date-filtered direct Chunk passages, but *does not return* unversioned
Entity.description strings. The Entity embedding index may still have been
built from the old description and influence *ranking*, so this gate by
itself does NOT make the entire entity-retrieval path temporally safe.

Remaining requirement (Issue #74): rebuild entity descriptions and Community
Reports from a verified date-specific corpus, carry per-claim/source-version
lineage, and rerun retrieval with a fresh embedding index. Direct dated
article evidence remains the standard for assessing legal answer grounding.
