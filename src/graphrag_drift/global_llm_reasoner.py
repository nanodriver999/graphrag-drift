from __future__ import annotations

import json
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
    """LLM-backed GraphRAG Global Search map/reduce reasoning."""

    generator: TextGenerator

    def map_community(self, query: str, report: CommunityReport) -> tuple[str, float]:
        prompt = (
            "You are a GraphRAG community analyst.\n"
            "Answer only from the supplied Community Report.\n"
            "Extract facts relevant to the user query and score their importance.\n"
            "Return ONLY a JSON object with keys answer and score. "
            "score must be an integer from 0 to 100. "
            "Use score 0 and an empty answer when the report is not relevant.\n\n"
            f"Query:\n{query}\n\n"
            f"Community ID:\n{report.id}\n\n"
            f"Community Report:\n{report.summary}\n"
        )
        raw = self.generator.generate(prompt).strip()

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            if raw == "NOT_RELEVANT" or not raw:
                return ("", 0.0)
            return (raw, report.score)

        if not isinstance(payload, dict):
            if raw == "NOT_RELEVANT" or not raw:
                return ("", 0.0)
            return (raw, report.score)

        answer = payload.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer) if answer is not None else ""
        answer = answer.strip()

        score = payload.get("score", 0)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        score_value = max(0.0, min(100.0, score_value)) / 100.0

        if not answer or score_value <= 0:
            return ("", 0.0)
        return (answer, score_value)

    def reduce_global(self, query: str, partials: list[tuple[str, float]]) -> str:
        ranked = sorted(partials, key=lambda item: item[1], reverse=True)
        useful = [
            (text, score)
            for text, score in ranked
            if score > 0 and text.strip() and text.strip() != "NOT_RELEVANT"
        ]

        if not useful:
            return "No relevant Community Report evidence was found."

        joined = "\n\n--- COMMUNITY PARTIAL ---\n\n".join(
            f"Importance Score: {score:.3f}\n{text}"
            for text, score in useful
        )
        prompt = (
            "You are a GraphRAG global answer synthesizer.\n"
            "Use only the supplied community partial answers.\n"
            "Prioritize higher importance scores when evidence overlaps.\n"
            "Synthesize a concise answer to the query.\n"
            "Do not introduce facts that are absent from the partial answers.\n\n"
            f"Query:\n{query}\n\n"
            f"Community partial answers:\n{joined}\n"
        )
        return self.generator.generate(prompt).strip()
