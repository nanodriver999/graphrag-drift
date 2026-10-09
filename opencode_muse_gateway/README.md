# OpenCode Muse Gateway

Portable OpenAI-compatible gateway for OpenCode Zen's Muse Spark 1.3 Contributor Free model.

**Copy this entire folder into another project.** It has no dependency on `graphrag-drift`.

## Architecture

```text
OpenAI-compatible client
  -> http://127.0.0.1:8000/v1/chat/completions
  -> gateway.py
  -> OpenCode HTTP session/message API :4096
  -> OpenCode Zen
  -> muse-spark-1.3-contributor-free
```

## Requirements

- Linux x64 or arm64
- Python 3.11+
- curl
- OpenCode Zen authentication configured in the environment

`install_opencode.sh` pins OpenCode v1.18.34 and verifies SHA-256.

## Quick start

```bash
cd opencode_muse_gateway
cp .env.example .env

# Edit .env if needed, then:
set -a
source .env
set +a

chmod +x install_opencode.sh start.sh stop.sh
./start.sh
```

The default endpoint is:

```text
http://127.0.0.1:8000/v1
```

Endpoints:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

Default model:

```text
muse-spark-1.3-contributor-free
```

## Use from another project

For an OpenAI-compatible client:

```text
base_url = http://127.0.0.1:8000/v1
model    = muse-spark-1.3-contributor-free
```

No shim API key is required by default because the gateway is intended for localhost use.
Do not expose port 8000 publicly without adding authentication or a protected reverse proxy.

A dependency-free client example is provided in `client_example.py`.

## Manual start

```bash
./install_opencode.sh

.opencode-bin/opencode serve --hostname 127.0.0.1 --port 4096

OPENCODE_URL=http://127.0.0.1:4096 \
OPENCODE_PROVIDER_ID=opencode \
OPENCODE_MODEL_ID=muse-spark-1.3-contributor-free \
python gateway.py --host 127.0.0.1 --port 8000
```

## Stop

```bash
./stop.sh
```

## Limitations

- streaming responses are not implemented;
- text chat-completions are translated to OpenCode session/message calls;
- OpenCode itself handles Zen authentication.
