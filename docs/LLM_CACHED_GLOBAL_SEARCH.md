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
