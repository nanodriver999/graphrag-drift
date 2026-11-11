# Direct legal source grounding in DRIFT benchmarks

A high answer-keyword score is **not** proof that a legal answer is correct.
For example, an LLM may mention `원자력안전법 제53조의3` without
retrieving the article as effective on the benchmark date.

The benchmark now records a distinct, conservative source-grounding signal:

- `direct_article_sources` are derived exclusively from retrieved
  `chunk-article:kr:law:...:effective:YYYYMMDD:art:...` hit IDs.
  Entity hits and LLM-generated DRIFT primers/action answers do not qualify.
- For benchmark cases with `expected_articles` (e.g. `["011435:53-3"]`),
  `direct_article_source_coverage` is the fraction of required article IDs
  retrieved from a version whose effective date is at or before
  `GRAPHRAG_AS_OF_DATE`.
- `future_only_expected_article_sources` lists expected articles seen
  exclusively in later-effective evidence (not acceptable for the date).
- `expected_article_mentions_without_direct_source` flags an article
  referenced in an answer even though the required date-effective direct
  article was not retrieved. **This is a review flag, not a claim of
  hallucination, contradiction, or a legal correctness judgment.**

If a legacy cached benchmark lacks `direct_article_sources`, it receives
`article_source_audit_status=source_provenance_not_recorded` rather than
an incorrect zero score. If the legal cutoff is unknown, no date-effective
source score is generated.

The initial gold article is `011435:53-3`, attached only to the primary
safety-manager-duties smoke case. Add other gold article mappings only after
verifying their true effective versions. Article presence in the source
does **not** prove that the answer accurately represents the provision; a
separate human or entailment-based legal audit remains necessary.

Strict comparison runs must be created with an immutable app revision
containing this evaluator; the existing pinned legacy comparison remains
unchanged.
