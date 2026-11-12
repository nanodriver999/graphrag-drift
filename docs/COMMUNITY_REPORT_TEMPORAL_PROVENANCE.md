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
