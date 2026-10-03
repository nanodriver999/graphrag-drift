from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .llm import DeterministicReasoner
from .models import CommunityReport


class TextGenerator(Protocol):
    """Minimal provider-neutral text generation interface."""

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class GlobalLLMReasoner(DeterministicReasoner):
    """Use an injected text generator for GraphRAG Global Search map/reduce.

    Local Search and DRIFT keep the existing deterministic behavior. This keeps
    the first LLM integration boundary limited to CommunityReport map/reduce.
    """

    generator: TextGenerator

    def map_community(self, query: str, report: CommunityReport) -> tuple[str, float]:
        prompt = (
            "You are a GraphRAG community analyst.\n"
            "Answer only from the supplied Community Report.\n"
            "Extract the facts that are relevant to the user query.\n"
            "If the report is not relevant, say NOT_RELEVANT.\n\n"
            f"Query:\n{query}\n\n"
            f"Community ID:\n{report.id}\n\n"
            f"Community Report:\n{report.summary}\n"
        )
        partial = self.generator.generate(prompt).strip()
        return (partial, report.score)

    def reduce_global(self, query: str, partials: list[tuple[str, float]]) -> str:
        ranked = sorted(partials, key=lambda item: item[1], reverse=True)
        useful = [text for text, _score in ranked if text.strip() and text.strip() != "NOT_RELEVANT"]

        if not useful:
            return "No relevant Community Report evidence was found."

        joined = "\n\n--- COMMUNITY PARTIAL ---\n\n".join(useful)
        prompt = (
            "You are a GraphRAG global answer synthesizer.\n"
            "Use only the supplied community partial answers.\n"
            "Synthesize a concise answer to the query.\n"
            "Do not introduce facts that are absent from the partial answers.\n\n"
            f"Query:\n{query}\n\n"
            f"Community partial answers:\n{joined}\n"
        )
        return self.generator.generate(prompt).strip()
