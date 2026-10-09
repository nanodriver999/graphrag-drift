#!/usr/bin/env bash
set -euo pipefail

VERSION="${OPENCODE_VERSION:-1.18.34}"
INSTALL_DIR="${OPENCODE_INSTALL_DIR:-$PWD/.opencode-bin}"
ARCHIVE="$INSTALL_DIR/opencode.tar.gz"
BIN="$INSTALL_DIR/opencode"

mkdir -p "$INSTALL_DIR"

if [ -x "$BIN" ]; then
  "$BIN" --version
  exit 0
fi

case "$(uname -m)" in
  x86_64|amd64)
    ASSET="opencode-linux-x64.tar.gz"
    SHA256="0f22479647226d1d2dd99595d20082ee7bda3870b62dc6a90b41efc1a71d7e9a"
    ;;
  aarch64|arm64)
    ASSET="opencode-linux-arm64.tar.gz"
    SHA256="bbdb3f00c2c51e42e315525233151309724226a8776da8e9145e3b0fa3d5310f"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

curl -fL --retry 3   "https://github.com/anomalyco/opencode/releases/download/v${VERSION}/${ASSET}"   -o "$ARCHIVE"

echo "$SHA256  $ARCHIVE" | sha256sum -c -
tar -xzf "$ARCHIVE" -C "$INSTALL_DIR"
chmod +x "$BIN"
"$BIN" --version
