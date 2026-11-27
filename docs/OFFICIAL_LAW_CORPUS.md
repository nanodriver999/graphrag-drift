# Date-effective whole-law source corpus

This is the first step in repairing the DRIFT corpus's missing 2026-effective
Korean Nuclear Safety Act source (law ID 011435). It is NOT a Neo4j import
or a rebuilt snapshot. Historical dumps, reports and runs remain unchanged.

## Official API contract

Official guide: https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsEfYdInfoGuide

The eflaw API documentation says omitting JO returns all articles. We send
target=eflaw, an exact MST and exact efYd. The secret OC is supplied using
LAW_GO_KR_OC, and must never be logged.

The downloader verifies returned law ID, effective date, and MST when returned.
It rejects duplicate/malformed article IDs, bad dates, wrong metadata, empty
eligible clauses, and missing required clauses. A later-effective article
is excluded from the accepted packets, with its identity and date recorded
in excluded_future_articles for explicit audit.

## Fetch one pinned whole-law version

Set the secret locally or as a private Actions secret. Example command:

    export LAW_GO_KR_OC="<credential>"
    python scripts/fetch_official_effective_law_corpus.py \
      --law-id 011435 --mst 286119 \
      --effective-date 2026-05-19 --as-of-date 2026-10-05 \
      --require-article 53-3 \
      --output-dir /tmp/official-law-011435-286119-20261005

The output is written to a NEW folder and will not overwrite an old corpus:

    official-law-011435-286119-20261005/
      source.xml
      manifest.json
      articles/
        article-000100.json
        article-005303.json
        ...

Each packet is marked downloaded_not_imported and references the SHA-256 of
the preserved original full XML. The JSON schema matches the earlier
single-article collector and may be validated by the PR #82 importer.

Verify saved artifacts offline, without the OC secret or any database:

    python scripts/fetch_official_effective_law_corpus.py \
      --verify-dir /tmp/official-law-011435-286119-20261005

The verifier re-parses the original XML and checks the entire manifest, all
article packets and the exact file set. Tampering/missing/extra files fail.

## Safety limitations and next step

- SHA-256 ties the JSON to the stored bytes; it is not an official digital
  signature by law.go.kr.
- The initial response limit is 4 MiB, matching the previous importer.
  Longer official XML is rejected rather than silently truncated.
- This imports no data. Other law versions need separate pinned fetches.
- Future-effective articles are quarantined; this step alone does not prove
  the complete legal corpus for all benchmark questions.
- The following PR must stage accepted packets on a disposable restored
  Neo4j copy, rebuild supported graph relationships and fulltext indexes,
  regenerate Leiden/Community Reports, and write a NEW snapshot SHA-256.
  A raw Chunk alone does not create MENTIONS or IN_COMMUNITY relationships.
