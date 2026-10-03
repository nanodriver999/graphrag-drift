from __future__ import annotations

from dataclasses import dataclass, field

from graphrag_drift.global_llm_reasoner import GlobalLLMReasoner
from graphrag_drift.models import CommunityReport


@dataclass
class FakeGenerator:
    responses: list[str]
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_global_llm_reasoner_maps_from_single_community_report() -> None:
    generator = FakeGenerator(responses=["Relevant duty: appoint a safety manager."])
    reasoner = GlobalLLMReasoner(generator=generator)
    report = CommunityReport(
        id="leiden-8",
        summary="Licensed users appoint a radiation safety manager.",
        score=0.9,
    )

    partial = reasoner.map_community("What are the safety-manager duties?", report)

    assert partial == ("Relevant duty: appoint a safety manager.", 0.9)
    assert "leiden-8" in generator.prompts[0]
    assert "Licensed users appoint a radiation safety manager." in generator.prompts[0]
    assert "What are the safety-manager duties?" in generator.prompts[0]


def test_global_llm_reasoner_reduces_only_useful_partials() -> None:
    generator = FakeGenerator(responses=["Combined grounded answer."])
    reasoner = GlobalLLMReasoner(generator=generator)

    answer = reasoner.reduce_global(
        "Summarize the duties.",
        [
            ("NOT_RELEVANT", 1.0),
            ("Operators must manage radiation dose.", 0.8),
            ("Safety managers must be appointed.", 0.7),
        ],
    )

    assert answer == "Combined grounded answer."
    prompt = generator.prompts[0]
    assert "Operators must manage radiation dose." in prompt
    assert "Safety managers must be appointed." in prompt
    assert "NOT_RELEVANT" not in prompt


def test_global_llm_reasoner_returns_fallback_when_nothing_is_relevant() -> None:
    generator = FakeGenerator(responses=[])
    reasoner = GlobalLLMReasoner(generator=generator)

    answer = reasoner.reduce_global(
        "unrelated query",
        [("NOT_RELEVANT", 1.0), ("", 0.5)],
    )

    assert answer == "No relevant Community Report evidence was found."
    assert generator.prompts == []


def test_global_llm_reasoner_keeps_existing_local_and_drift_behavior() -> None:
    generator = FakeGenerator(responses=[])
    reasoner = GlobalLLMReasoner(generator=generator)

    local = reasoner.answer_local("q", [])
    drift = reasoner.reduce_drift("q", [])

    assert local == "LOCAL[q]: "
    assert drift == "DRIFT[q]: "
