#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${OPENCODE_INSTALL_DIR:-$ROOT/.opencode-bin}"
OPENCODE_BIN="${OPENCODE_BIN:-$BIN_DIR/opencode}"
RUN_DIR="${OPENCODE_GATEWAY_RUN_DIR:-$ROOT/.run}"

mkdir -p "$RUN_DIR"

: "${OPENCODE_HOST:=127.0.0.1}"
: "${OPENCODE_PORT:=4096}"
: "${SHIM_HOST:=127.0.0.1}"
: "${SHIM_PORT:=8000}"
: "${OPENCODE_PROVIDER_ID:=opencode}"
: "${OPENCODE_MODEL_ID:=muse-spark-1.3-contributor-free}"
: "${OPENCODE_REQUEST_TIMEOUT:=180}"

if [ ! -x "$OPENCODE_BIN" ]; then
  OPENCODE_INSTALL_DIR="$BIN_DIR" "$ROOT/install_opencode.sh"
fi

nohup "$OPENCODE_BIN" serve   --hostname "$OPENCODE_HOST"   --port "$OPENCODE_PORT"   > "$RUN_DIR/opencode.log" 2>&1 &
echo $! > "$RUN_DIR/opencode.pid"

for _ in $(seq 1 60); do
  if curl -fsS "http://$OPENCODE_HOST:$OPENCODE_PORT/global/health" >/dev/null; then
    break
  fi
  sleep 1
done

export OPENCODE_URL="http://$OPENCODE_HOST:$OPENCODE_PORT"
nohup python "$ROOT/gateway.py"   --host "$SHIM_HOST"   --port "$SHIM_PORT"   --opencode-url "$OPENCODE_URL"   --provider "$OPENCODE_PROVIDER_ID"   --model "$OPENCODE_MODEL_ID"   --timeout "$OPENCODE_REQUEST_TIMEOUT"   > "$RUN_DIR/shim.log" 2>&1 &
echo $! > "$RUN_DIR/shim.pid"

for _ in $(seq 1 60); do
  if curl -fsS "http://$SHIM_HOST:$SHIM_PORT/health" >/dev/null; then
    echo "OpenAI-compatible endpoint: http://$SHIM_HOST:$SHIM_PORT/v1"
    echo "Model: $OPENCODE_MODEL_ID"
    exit 0
  fi
  sleep 1
done

cat "$RUN_DIR/opencode.log" >&2 || true
cat "$RUN_DIR/shim.log" >&2 || true
exit 1
