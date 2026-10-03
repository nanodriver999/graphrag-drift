# LLM-backed cached Global Search runtime

This runtime composes the already versioned Community Report cache with the
provider-neutral Global Search LLM reasoner.

## Environment

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000"
export OPENAI_COMPATIBLE_MODEL="my-model"
export OPENAI_COMPATIBLE_API_KEY=""
```

Optional:

```bash
export OPENAI_COMPATIBLE_TIMEOUT_SECONDS="60"
export OPENAI_COMPATIBLE_TEMPERATURE="0"
```

`OPENAI_COMPATIBLE_API_KEY` may be omitted for local endpoints such as a
vLLM server that does not require authentication.

## Composition

```text
current data dump SHA
  -> matching Community Report cache manifest
  -> FileCommunityReportStore
  -> CachedCommunityReportRetriever
  -> GlobalLLMReasoner
  -> OpenAICompatibleTextGenerator
  -> GraphRAGEngine
```

This module only constructs the runtime. An executable CLI and live-provider
smoke test are separate tasks.


## Run

After setting the environment variables above:

```bash
python scripts/run_llm_cached_global_search.py \
  "방사선안전관리자의 의무는 무엇인가?" \
  --data-repo ../graphrag-drift-data \
  --top-k 3
```

The command performs:

```text
query
  -> cached Community Report retrieval
  -> LLM map for each selected community
  -> filter NOT_RELEVANT partials
  -> LLM global reduce
  -> JSON output
```

The data repository's Community Reports remain the grounding source. The LLM is
used to select relevant facts from those cached reports and synthesize the final
Global Search answer.
