from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_drift.opencode_model_selector import (
    OpenCodeClient,
    select_working_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a currently working zero-cost model exposed by OpenCode."
    )
    parser.add_argument("--opencode-url", default="http://127.0.0.1:4096")
    parser.add_argument(
        "--preferred-model",
        action="append",
        dest="preferred_models",
        default=[],
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--github-env", default=None)
    args = parser.parse_args()

    client = OpenCodeClient(args.opencode_url, args.timeout)
    payload = client.request("/provider")
    if not isinstance(payload, dict):
        raise RuntimeError("OpenCode /provider returned a non-object response")

    selected, attempts = select_working_model(
        client,
        payload,
        preferred_model_ids=args.preferred_models,
    )
    result = {
        "format": "graphrag-drift-opencode-model-selection-v1",
        "selected": selected.serialize(),
        "attempts": attempts,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    if args.github_env:
        with Path(args.github_env).open("a", encoding="utf-8") as out:
            out.write(f"OPENCODE_PROVIDER_ID={selected.provider_id}\n")
            out.write(f"OPENCODE_MODEL_ID={selected.model_id}\n")
            out.write(f"OPENAI_COMPATIBLE_MODEL={selected.model_id}\n")


if __name__ == "__main__":
    main()
