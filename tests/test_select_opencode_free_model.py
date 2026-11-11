from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.select_opencode_free_model import (
    ModelCandidate,
    select_working_model,
    zero_cost_candidates,
)


def provider_payload() -> dict:
    return {
        "connected": ["opencode", "other"],
        "all": [
            {
                "id": "opencode",
                "models": {
                    "broken-free": {
                        "name": "Broken Free",
                        "cost": {"input": 0, "output": 0},
                    },
                    "working-free": {
                        "name": "Working Free",
                        "cost": {"input": 0, "output": 0},
                    },
                    "paid": {
                        "name": "Paid",
                        "cost": {"input": 1, "output": 1},
                    },
                },
            },
            {
                "id": "disconnected",
                "models": {
                    "free": {
                        "name": "Disconnected Free",
                        "cost": {"input": 0, "output": 0},
                    }
                },
            },
        ],
    }


@dataclass
class FakeClient:
    broken_models: set[str]

    def request(self, path: str, *, method: str = "GET", body: dict | None = None):
        if path == "/session":
            return {"id": "session-1"}
        if path.startswith("/session/") and path.endswith("/message"):
            model_id = body["model"]["modelID"]
            if model_id in self.broken_models:
                raise RuntimeError("upstream unavailable")
            return {
                "parts": [
                    {"type": "text", "text": "OPENCODE_MODEL_OK"},
                ]
            }
        raise AssertionError(path)


def test_zero_cost_candidates_filters_connected_and_paid_models() -> None:
    candidates = zero_cost_candidates(provider_payload())

    assert [(item.provider_id, item.model_id) for item in candidates] == [
        ("opencode", "broken-free"),
        ("opencode", "working-free"),
    ]


def test_zero_cost_candidates_respects_preferred_order() -> None:
    candidates = zero_cost_candidates(
        provider_payload(),
        preferred_model_ids=["working-free", "broken-free"],
    )

    assert [item.model_id for item in candidates] == [
        "working-free",
        "broken-free",
    ]


def test_selector_falls_back_after_failed_zero_cost_model() -> None:
    selected, attempts = select_working_model(
        FakeClient(broken_models={"broken-free"}),
        provider_payload(),
        preferred_model_ids=["broken-free", "working-free"],
    )

    assert selected == ModelCandidate(
        provider_id="opencode",
        model_id="working-free",
        name="Working Free",
    )
    assert attempts[0]["model_id"] == "broken-free"
    assert "error" in attempts[0]
    assert attempts[1]["status"] == "selected"


def test_selector_fails_when_all_zero_cost_models_are_unavailable() -> None:
    with pytest.raises(RuntimeError, match="No zero-cost OpenCode model passed"):
        select_working_model(
            FakeClient(broken_models={"broken-free", "working-free"}),
            provider_payload(),
        )
