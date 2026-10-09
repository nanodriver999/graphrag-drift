# OpenCode + Muse Spark 1.3 Contributor Free

The project can use OpenCode in headless HTTP server mode as the model gateway for
the existing OpenAI-compatible GraphRAG runtime.

## Model

OpenCode Zen currently exposes:

```text
opencode/muse-spark-1.3-contributor-free
```

The model is free for a limited period. It is a contributor tier model, so prompts
and completions may be used to improve future Meta models.

OpenCode Zen still requires a Zen account/API key.

## 1. Start OpenCode

```bash
export OPENCODE_SERVER_PASSWORD="choose-a-local-password"

opencode serve \
  --hostname 127.0.0.1 \
  --port 4096
```

Configure OpenCode Zen authentication using the normal OpenCode auth flow before
sending real model requests.

## 2. Start the OpenAI-compatible shim

```bash
export OPENCODE_URL="http://127.0.0.1:4096"
export OPENCODE_PROVIDER_ID="opencode"
export OPENCODE_MODEL_ID="muse-spark-1.3-contributor-free"

python opencode_muse_gateway/gateway.py \
  --host 127.0.0.1 \
  --port 8000
```

The shim exposes:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

## 3. Point GraphRAG at the shim

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://127.0.0.1:8000"
export OPENAI_COMPATIBLE_MODEL="muse-spark-1.3-contributor-free"
unset OPENAI_COMPATIBLE_API_KEY
```

Then run the existing command:

```bash
python scripts/run_llm_cached_global_search.py \
  "방사선안전관리자의 의무는 무엇인가?" \
  --data-repo ../graphrag-drift-data \
  --top-k 3
```

The resulting path is:

```text
GraphRAG
  -> OpenAI-compatible /v1/chat/completions
  -> local shim
  -> OpenCode HTTP session/message API
  -> OpenCode Zen
  -> Muse Spark 1.3 Contributor Free
```

## Sandbox note

A sandbox without outbound network access can still validate the local OpenCode
server and shim translation using a local provider. A real Muse request additionally
requires outbound access to the OpenCode Zen API.


## Portable folder

The OpenCode/Muse gateway is now self-contained under:

```text
opencode_muse_gateway/
```

Copy that directory into another project to reuse the same OpenAI-compatible endpoint.
See `opencode_muse_gateway/README.md` for standalone install/start instructions.

`scripts/opencode_openai_shim.py` remains only as a backward-compatible entry point.
