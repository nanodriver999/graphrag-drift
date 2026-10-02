# Cached Global Search

The current GraphRAG prototype can use the versioned Community Reports stored in
`graphrag-drift-data` without regenerating those reports.

## Prerequisites

Check out both repositories next to each other:

```text
workspace/
  graphrag-drift/
  graphrag-drift-data/
```

The data repository must contain a complete cache for the current dump:

```bash
cd graphrag-drift-data
python scripts/check_community_report_cache.py --require-complete
```

## Run Global Search

From the application repository:

```bash
python scripts/run_cached_global_search.py \
  "방사선안전관리자의 의무는 무엇인가?" \
  --data-repo ../graphrag-drift-data \
  --top-k 3
```

The CLI:

1. reads `data/manifest.json` from the data repository;
2. derives the active dump SHA prefix;
3. opens the matching cached Community Report manifest;
4. ranks cached reports for the query;
5. passes them through the existing `GraphRAGEngine.global_search()` flow.

## Current reasoner

This CLI intentionally uses the project's existing deterministic reasoner.

Therefore it demonstrates the cached GraphRAG Global Search data path, not a final
production-quality LLM answer. Adding an LLM-backed map/reduce reasoner is a
separate task.
