#!/usr/bin/env python3
"""Backward-compatible entry point for the reusable gateway."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencode_muse_gateway.gateway import (  # noqa: E402,F401
    Handler,
    OpenAIShimServer,
    OpenCodeClient,
    create_server,
    main,
)

if __name__ == "__main__":
    main()
