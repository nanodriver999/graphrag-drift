# DRIFT profile benchmark provenance

The DRIFT profile comparison supports an optional, strict provenance identity.
Use it for **new** published comparisons so a resumed benchmark cannot silently
reuse measurements from another model, application revision, legal effective date,
or Neo4j snapshot.

## Pinned comparison

```bash
export GRAPHRAG_AS_OF_DATE=2026-10-05
export GRAPHRAG_SNAPSHOT_SHA256="$(sha256sum snapshots/community-ready/neo4j.dump | cut -d ' ' -f 1)"
export GRAPHRAG_APP_REVISION="$(git -C app rev-parse HEAD)"
# OPENAI_COMPATIBLE_MODEL and OPENAI_COMPATIBLE_MODEL_FALLBACKS
# must resolve to the same values on every resumed invocation.

python app/scripts/run_drift_profile_comparison.py \
  --cases app/benchmarks/legal_smoke.jsonl \
  --progress-json benchmark_runs/2026-11-11-pinned-001/progress.json \
  --checkpoint-db benchmark_runs/2026-11-11-pinned-001/checkpoint.sqlite \
  --run-id 2026-11-11-pinned-001 \
  --profiles economy microsoft-like \
  --baseline-profile economy
```

The identity includes:

- actual configured primary and fallback model IDs (not per-request fallback usage)
- source Neo4j snapshot SHA-256
- application code revision
- `GRAPHRAG_AS_OF_DATE` (law effective-date cutoff)
- model temperature and maximum in-flight provider calls

The identity is written to `progress.json` and the final comparison JSON.
A mismatch raises an error **before** Neo4j is opened or a LangGraph thread is
resumed. The checkpoint thread fingerprint also incorporates the identity.

## Existing legacy runs

Existing progress files written before provenance pinning do **not** have an
identity; they remain readable in legacy mode, but cannot be used as input to a
new pinned run. Create a fresh run ID and empty progress/checkpoint directory
rather than relabeling measurements from an earlier run.

This guards *configured* model identity, not the model actually used for an
individual fallback request. When an API fallback chain is enabled, those
results are not strictly single-model comparisons. Prefer a fixed provider/model
for scientific comparisons and retain the selection logs as artifacts.
