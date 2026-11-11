from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opencode_muse_gateway.gateway import OpenCodeClient


@dataclass(frozen=True)
class ModelCandidate:
    provider_id: str
    model_id: str
    name: str

    def serialize(self) -> dict[str, str]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "name": self.name,
        }


def zero_cost_candidates(
    payload: dict[str, Any],
    *,
    preferred_model_ids: list[str] | None = None,
) -> list[ModelCandidate]:
    connected = {
        str(item)
        for item in payload.get("connected", [])
        if isinstance(item, str)
    }
    providers = payload.get("all", [])
    if not isinstance(providers, list):
        return []

    rows: list[ModelCandidate] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_id = str(provider.get("id") or "")
        if not provider_id or provider_id not in connected:
            continue
        models = provider.get("models", {})
        if not isinstance(models, dict):
            continue

        for model_id, metadata in models.items():
            if not isinstance(metadata, dict):
                continue
            cost = metadata.get("cost", {})
            if not isinstance(cost, dict):
                continue
            if cost.get("input") != 0 or cost.get("output") != 0:
                continue
            rows.append(
                ModelCandidate(
                    provider_id=provider_id,
                    model_id=str(model_id),
                    name=str(metadata.get("name") or model_id),
                )
            )

    preferred = preferred_model_ids or []
    preferred_order = {model_id: index for index, model_id in enumerate(preferred)}
    rows.sort(
        key=lambda item: (
            0 if item.model_id in preferred_order else 1,
            preferred_order.get(item.model_id, 10_000),
            item.provider_id,
            item.model_id,
        )
    )
    return rows


def probe_candidate(
    client: OpenCodeClient,
    candidate: ModelCandidate,
    *,
    prompt: str = "Reply with exactly OPENCODE_MODEL_OK.",
) -> str:
    session = client.request(
        "/session",
        method="POST",
        body={"title": "Zero-cost model availability probe"},
    )
    if not isinstance(session, dict) or not session.get("id"):
        raise RuntimeError("OpenCode did not return a session id")

    result = client.request(
        f"/session/{session['id']}/message",
        method="POST",
        body={
            "model": {
                "providerID": candidate.provider_id,
                "modelID": candidate.model_id,
            },
            "parts": [{"type": "text", "text": prompt}],
            "tools": {},
        },
    )
    if not isinstance(result, dict):
        raise RuntimeError("OpenCode model probe returned a non-object response")

    content = "\n".join(
        str(part.get("text") or "")
        for part in result.get("parts", [])
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()
    if not content:
        raise RuntimeError("OpenCode model probe returned empty text")
    return content


def select_working_model(
    client: OpenCodeClient,
    payload: dict[str, Any],
    *,
    preferred_model_ids: list[str] | None = None,
) -> tuple[ModelCandidate, list[dict[str, str]]]:
    candidates = zero_cost_candidates(
        payload,
        preferred_model_ids=preferred_model_ids,
    )
    if not candidates:
        raise RuntimeError("No connected zero-cost OpenCode models were found")

    failures: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            response = probe_candidate(client, candidate)
        except Exception as exc:
            failures.append(
                {
                    **candidate.serialize(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        return candidate, [
            *failures,
            {
                **candidate.serialize(),
                "status": "selected",
                "response": response,
            },
        ]

    details = "; ".join(
        f"{item['provider_id']}/{item['model_id']}: {item['error']}"
        for item in failures
    )
    raise RuntimeError(f"No zero-cost OpenCode model passed the live probe: {details}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a currently working zero-cost model exposed by OpenCode."
    )
    parser.add_argument(
        "--opencode-url",
        default="http://127.0.0.1:4096",
    )
    parser.add_argument(
        "--preferred-model",
        action="append",
        dest="preferred_models",
        default=[],
        help="Preferred model ID. Repeat to specify fallback order.",
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
        env_path = Path(args.github_env)
        with env_path.open("a", encoding="utf-8") as out:
            out.write(f"OPENCODE_PROVIDER_ID={selected.provider_id}\n")
            out.write(f"OPENCODE_MODEL_ID={selected.model_id}\n")
            out.write(f"OPENAI_COMPATIBLE_MODEL={selected.model_id}\n")


if __name__ == "__main__":
    main()
