from __future__ import annotations

from dataclasses import dataclass, field

from graphrag_drift.models import SearchHit
from graphrag_drift.search_llm_reasoner import GraphRAGLLMReasoner


@dataclass
class FakeGenerator:
    responses: list[str]
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_local_answer_is_grounded_in_retrieved_evidence() -> None:
    generator = FakeGenerator(responses=["Grounded local answer."])
    reasoner = GraphRAGLLMReasoner(generator=generator)
    hits = [SearchHit("e1", "Safety managers must be appointed before use.", 0.9)]

    answer = reasoner.answer_local("When is a safety manager appointed?", hits)

    assert answer == "Grounded local answer."
    prompt = generator.prompts[0]
    assert "local-search answer synthesizer" in prompt
    assert "Safety managers must be appointed before use." in prompt


def test_drift_followups_parse_json_and_are_bounded() -> None:
    generator = FakeGenerator(
        responses=['["delegated safety work requirements", "agent registration requirements", "extra"]']
    )
    reasoner = GraphRAGLLMReasoner(generator=generator, max_followups=2)
    hits = [SearchHit("e1", "Registered agents may perform delegated work.", 0.8)]

    followups = reasoner.generate_followups("Explain delegated safety work.", hits, depth=0)

    assert followups == [
        "delegated safety work requirements",
        "agent registration requirements",
    ]
    assert "Return ONLY a JSON array of strings." in generator.prompts[0]


def test_drift_followups_stop_on_invalid_json() -> None:
    generator = FakeGenerator(responses=["not json"])
    reasoner = GraphRAGLLMReasoner(generator=generator)

    followups = reasoner.generate_followups(
        "q",
        [SearchHit("e1", "evidence", 1.0)],
        depth=0,
    )

    assert followups == []


def test_drift_followups_stop_at_depth_two_without_model_call() -> None:
    generator = FakeGenerator(responses=[])
    reasoner = GraphRAGLLMReasoner(generator=generator)

    followups = reasoner.generate_followups(
        "q",
        [SearchHit("e1", "evidence", 1.0)],
        depth=2,
    )

    assert followups == []
    assert generator.prompts == []


def test_drift_reduce_deduplicates_evidence_in_prompt() -> None:
    generator = FakeGenerator(responses=["Final DRIFT answer."])
    reasoner = GraphRAGLLMReasoner(generator=generator)
    hits = [
        SearchHit("e1", "same evidence", 0.9),
        SearchHit("e1", "same evidence", 0.9),
        SearchHit("e2", "second evidence", 0.8),
    ]

    answer = reasoner.reduce_drift("q", hits)

    assert answer == "Final DRIFT answer."
    prompt = generator.prompts[0]
    assert prompt.count("[e1]") == 1
    assert prompt.count("[e2]") == 1
