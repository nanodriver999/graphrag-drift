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


def test_global_llm_reasoner_keeps_plain_text_map_fallback() -> None:
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


def test_global_llm_reasoner_uses_structured_map_relevance_score() -> None:
    generator = FakeGenerator(
        responses=['{"answer":"Relevant legal duty","score":82}']
    )
    reasoner = GlobalLLMReasoner(generator=generator)
    report = CommunityReport("c1", "report", 0.2)

    partial = reasoner.map_community("query", report)

    assert partial == ("Relevant legal duty", 0.82)
    assert "score must be an integer from 0 to 100" in generator.prompts[0]


def test_global_llm_reasoner_maps_zero_score_to_irrelevant() -> None:
    generator = FakeGenerator(
        responses=['{"answer":"should be ignored","score":0}']
    )
    reasoner = GlobalLLMReasoner(generator=generator)

    assert reasoner.map_community(
        "query",
        CommunityReport("c1", "report", 1.0),
    ) == ("", 0.0)


def test_global_llm_reasoner_reduces_only_useful_partials_by_score() -> None:
    generator = FakeGenerator(responses=["Combined grounded answer."])
    reasoner = GlobalLLMReasoner(generator=generator)

    answer = reasoner.reduce_global(
        "Summarize the duties.",
        [
            ("ignored", 0.0),
            ("Lower priority evidence.", 0.4),
            ("Higher priority evidence.", 0.9),
        ],
    )

    assert answer == "Combined grounded answer."
    prompt = generator.prompts[0]
    assert "ignored" not in prompt
    assert prompt.index("Higher priority evidence.") < prompt.index("Lower priority evidence.")
    assert "Importance Score: 0.900" in prompt


def test_global_llm_reasoner_returns_fallback_when_nothing_is_relevant() -> None:
    generator = FakeGenerator(responses=[])
    reasoner = GlobalLLMReasoner(generator=generator)

    answer = reasoner.reduce_global(
        "unrelated query",
        [("", 0.0), ("NOT_RELEVANT", 0.0)],
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
