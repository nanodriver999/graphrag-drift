from __future__ import annotations

from dataclasses import dataclass

import pytest

from graphrag_drift.opencode_model_selector import (
    ModelCandidate,
    select_working_model,
    select_working_models,
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


@dataclass
class WorkloadProbeClient:
    broken_models: set[str]

    def request(self, path: str, *, method: str = "GET", body: dict | None = None):
        if path == "/session":
            return {"id": "session-1"}
        if path.startswith("/session/") and path.endswith("/message"):
            model_id = body["model"]["modelID"]
            if model_id in self.broken_models:
                raise TimeoutError("workload timeout")
            prompt = body["parts"][0]["text"]
            if "GraphRAG DRIFT-style" in prompt:
                return {
                    "parts": [
                        {
                            "type": "text",
                            "text": (
                                '{"intermediate_answer":"grounded","score":80,'
                                '"follow_up_queries":["q1","q2"]}'
                            ),
                        }
                    ]
                }
            return {"parts": [{"type": "text", "text": "OPENCODE_MODEL_OK"}]}
        raise AssertionError(path)


def test_selector_workload_probe_returns_viable_fallback_order() -> None:
    payload = provider_payload()
    selected, attempts = select_working_models(
        WorkloadProbeClient(broken_models=set()),
        payload,
        preferred_model_ids=["working-free", "broken-free"],
        max_models=2,
        workload_probe=True,
    )

    assert [item.model_id for item in selected] == [
        "working-free",
        "broken-free",
    ]
    assert selected[0].qualified_id == "opencode/working-free"
    selected_attempt = next(
        row for row in attempts if row["model_id"] == "working-free"
    )
    assert selected_attempt["status"] == "selected"
    assert "grounded" in selected_attempt["workload_response"]


def test_selector_skips_model_that_times_out_on_workload_probe() -> None:
    selected, attempts = select_working_models(
        WorkloadProbeClient(broken_models={"working-free"}),
        provider_payload(),
        preferred_model_ids=["working-free", "broken-free"],
        max_models=1,
        workload_probe=True,
    )

    assert selected[0].model_id == "broken-free"
    assert attempts[0]["model_id"] == "working-free"
    assert "TimeoutError" in attempts[0]["error"]



def test_required_zero_cost_model_is_selected_without_alternatives() -> None:
    selected, attempts = select_working_models(
        WorkloadProbeClient(broken_models=set()),
        provider_payload(),
        required_model="opencode/working-free",
        max_models=3,
        workload_probe=True,
    )
    assert [item.qualified_id for item in selected] == ["opencode/working-free"]
    assert len(attempts) == 1


def test_required_model_rejects_disconnected_paid_and_unknown_models() -> None:
    for required in [
        "opencode/paid",
        "disconnected/free",
        "opencode/missing",
    ]:
        with pytest.raises(RuntimeError, match="Required zero-cost"):
            select_working_models(
                WorkloadProbeClient(broken_models=set()),
                provider_payload(),
                required_model=required,
            )


def test_required_model_does_not_fallback_when_probe_fails() -> None:
    with pytest.raises(RuntimeError, match="opencode/working-free"):
        select_working_models(
            WorkloadProbeClient(broken_models={"working-free"}),
            provider_payload(),
            required_model="opencode/working-free",
            preferred_model_ids=["broken-free"],
        )
