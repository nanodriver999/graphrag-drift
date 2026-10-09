#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${OPENCODE_GATEWAY_RUN_DIR:-$ROOT/.run}"

for name in shim opencode; do
  pid_file="$RUN_DIR/$name.pid"
  if [ -f "$pid_file" ]; then
    kill "$(cat "$pid_file")" >/dev/null 2>&1 || true
    rm -f "$pid_file"
  fi
done
