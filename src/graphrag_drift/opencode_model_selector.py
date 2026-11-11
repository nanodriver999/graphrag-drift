from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any


WORKLOAD_PROBE_PROMPT = """You are testing whether an LLM can handle a GraphRAG DRIFT-style reasoning request.
Read the evidence below and return ONLY valid JSON with keys:
intermediate_answer (string), score (integer 0-100), follow_up_queries (array of strings).

Question: What duties and delegation relationships are described for a radiation safety manager?

Evidence:
- A radiation safety manager is appointed before regulated use begins and the appointment is reported.
- Some radiation-safety-management work may be delegated to a registered service provider.
- When the responsible person is absent, a qualified substitute may need to be designated.
- Inspection and compliance obligations can depend on the applicable statute and effective date.

Use only this evidence. Include 2 concise follow-up queries.
"""


class OpenCodeClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> Any:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=payload,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            raw = response.read()
            return json.loads(raw) if raw else None


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

    @property
    def qualified_id(self) -> str:
        return f"{self.provider_id}/{self.model_id}"


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
    title: str = "Zero-cost model availability probe",
) -> str:
    session = client.request(
        "/session",
        method="POST",
        body={"title": title},
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


def _validate_workload_probe(response: str) -> None:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("workload probe must return a JSON object")
    if not isinstance(payload.get("intermediate_answer"), str):
        raise ValueError("workload probe intermediate_answer must be a string")
    if not isinstance(payload.get("score"), int):
        raise ValueError("workload probe score must be an integer")
    followups = payload.get("follow_up_queries")
    if not isinstance(followups, list) or len(followups) < 2:
        raise ValueError("workload probe must return at least two follow-up queries")


def select_working_models(
    client: OpenCodeClient,
    payload: dict[str, Any],
    *,
    preferred_model_ids: list[str] | None = None,
    max_models: int = 3,
    workload_probe: bool = True,
) -> tuple[list[ModelCandidate], list[dict[str, str]]]:
    candidates = zero_cost_candidates(
        payload,
        preferred_model_ids=preferred_model_ids,
    )
    if not candidates:
        raise RuntimeError("No connected zero-cost OpenCode models were found")
    if max_models < 1:
        raise ValueError("max_models must be >= 1")

    selected: list[ModelCandidate] = []
    attempts: list[dict[str, str]] = []

    for candidate in candidates:
        try:
            quick = probe_candidate(client, candidate)
            workload = ""
            if workload_probe:
                workload = probe_candidate(
                    client,
                    candidate,
                    prompt=WORKLOAD_PROBE_PROMPT,
                    title="GraphRAG workload capability probe",
                )
                _validate_workload_probe(workload)
        except Exception as exc:
            attempts.append(
                {
                    **candidate.serialize(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        selected.append(candidate)
        attempts.append(
            {
                **candidate.serialize(),
                "status": "viable",
                "quick_response": quick,
                "workload_response": workload,
            }
        )
        if len(selected) >= max_models:
            break

    if not selected:
        details = "; ".join(
            f"{item['provider_id']}/{item['model_id']}: {item['error']}"
            for item in attempts
            if "error" in item
        )
        raise RuntimeError(
            f"No zero-cost OpenCode model passed the workload probe: {details}"
        )

    attempts[-1 if len(selected) == 1 else next(
        index for index, row in enumerate(attempts)
        if row.get("provider_id") == selected[0].provider_id
        and row.get("model_id") == selected[0].model_id
    )]["status"] = "selected"
    return selected, attempts


def select_working_model(
    client: OpenCodeClient,
    payload: dict[str, Any],
    *,
    preferred_model_ids: list[str] | None = None,
    workload_probe: bool = False,
) -> tuple[ModelCandidate, list[dict[str, str]]]:
    selected, attempts = select_working_models(
        client,
        payload,
        preferred_model_ids=preferred_model_ids,
        max_models=1,
        workload_probe=workload_probe,
    )
    return selected[0], attempts
