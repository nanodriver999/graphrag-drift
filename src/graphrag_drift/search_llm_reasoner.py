from __future__ import annotations

import json
from dataclasses import dataclass

from .global_llm_reasoner import GlobalLLMReasoner
from .models import SearchHit


def _format_evidence(evidence: list[SearchHit], *, limit: int = 20) -> str:
    unique: dict[str, SearchHit] = {}
    for hit in evidence:
        unique.setdefault(hit.id, hit)

    rows: list[str] = []
    for hit in list(unique.values())[:limit]:
        text = hit.text.strip()
        if len(text) > 6000:
            text = text[:6000] + "..."
        rows.append(f"[{hit.id}] score={hit.score:.4f}\n{text}")
    return "\n\n--- EVIDENCE ---\n\n".join(rows)


@dataclass
class GraphRAGLLMReasoner(GlobalLLMReasoner):
    """LLM-backed reasoner for Local, Global, and DRIFT GraphRAG modes."""

    max_followups: int = 2

    def answer_local(self, query: str, hits: list[SearchHit]) -> str:
        if not hits:
            return "No relevant local evidence was found."

        prompt = (
            "You are a GraphRAG local-search answer synthesizer.\n"
            "Answer only from the supplied retrieved evidence.\n"
            "Be concise and preserve important qualifications.\n"
            "Do not introduce facts absent from the evidence.\n"
            "If the evidence is insufficient, say so explicitly.\n\n"
            f"Query:\n{query}\n\n"
            f"Retrieved evidence:\n{_format_evidence(hits)}\n"
        )
        return self.generator.generate(prompt).strip()

    def generate_followups(
        self,
        query: str,
        evidence: list[SearchHit],
        *,
        depth: int,
    ) -> list[str]:
        if depth >= 2 or not evidence:
            return []

        prompt = (
            "You are planning the next DRIFT GraphRAG retrieval step.\n"
            "Based only on the original query and current evidence, propose at most "
            f"{self.max_followups} short follow-up search queries that would fill the "
            "most important missing information.\n"
            "Return ONLY a JSON array of strings. Return [] if no further search is needed.\n\n"
            f"Original query:\n{query}\n\n"
            f"Current evidence:\n{_format_evidence(evidence)}\n"
        )
        raw = self.generator.generate(prompt).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(payload, list):
            return []

        followups: list[str] = []
        for item in payload:
            if isinstance(item, str):
                item = item.strip()
                if item and item not in followups:
                    followups.append(item)
            if len(followups) >= self.max_followups:
                break
        return followups

    def reduce_drift(self, query: str, evidence: list[SearchHit]) -> str:
        if not evidence:
            return "No relevant DRIFT evidence was found."

        prompt = (
            "You are a DRIFT GraphRAG answer synthesizer.\n"
            "Answer the original query using only the accumulated evidence from iterative retrieval.\n"
            "Resolve overlaps, avoid duplication, and preserve uncertainty where evidence is incomplete.\n"
            "Do not introduce facts absent from the evidence.\n\n"
            f"Original query:\n{query}\n\n"
            f"Accumulated evidence:\n{_format_evidence(evidence)}\n"
        )
        return self.generator.generate(prompt).strip()
