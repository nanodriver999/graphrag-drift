from __future__ import annotations

import json
from dataclasses import dataclass

from .global_llm_reasoner import GlobalLLMReasoner
from .models import CommunityReport, DriftActionResult, DriftPrimerResult, SearchHit


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


def _format_reports(reports: list[CommunityReport], *, limit: int = 20) -> str:
    rows: list[str] = []
    for report in reports[:limit]:
        text = report.summary.strip()
        if len(text) > 6000:
            text = text[:6000] + "..."
        rows.append(f"[{report.id}] score={report.score:.4f}\n{text}")
    return "\n\n--- COMMUNITY REPORT ---\n\n".join(rows)


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

    def prime_drift(
        self,
        query: str,
        reports: list[CommunityReport],
    ) -> DriftPrimerResult:
        if not reports:
            return DriftPrimerResult("", 0, [])

        prompt = (
            "You are the primer phase of DRIFT GraphRAG.\n"
            "Use the top-ranked Community Reports to form a broad intermediate answer "
            "and propose focused follow-up queries for later Local Search.\n"
            "Use only the supplied reports for factual claims.\n"
            "Return ONLY a JSON object with keys intermediate_answer, score, "
            "follow_up_queries. score must be an integer from 0 to 100. "
            f"follow_up_queries must contain at most {self.max_followups} short strings.\n\n"
            f"Query:\n{query}\n\n"
            f"Community Reports:\n{_format_reports(reports)}\n"
        )
        raw = self.generator.generate(prompt).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            best = max(reports, key=lambda item: item.score)
            return DriftPrimerResult(
                intermediate_answer=best.summary.strip(),
                score=max(0, min(100, round(best.score * 100))),
                follow_up_queries=[],
            )

        if not isinstance(payload, dict):
            payload = {}

        answer = payload.get("intermediate_answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = max(reports, key=lambda item: item.score).summary

        score = payload.get("score", 0)
        if not isinstance(score, int):
            try:
                score = int(score)
            except (TypeError, ValueError):
                score = 0
        score = max(0, min(100, score))

        raw_followups = payload.get("follow_up_queries", [])
        followups: list[str] = []
        if isinstance(raw_followups, list):
            for item in raw_followups:
                if isinstance(item, str):
                    item = item.strip()
                    if item and item not in followups:
                        followups.append(item)
                if len(followups) >= self.max_followups:
                    break

        return DriftPrimerResult(
            intermediate_answer=answer.strip(),
            score=score,
            follow_up_queries=followups,
        )

    def answer_drift_action(
        self,
        global_query: str,
        action_query: str,
        hits: list[SearchHit],
        *,
        depth: int,
    ) -> DriftActionResult:
        if not hits:
            return DriftActionResult("", 0, [])

        prompt = (
            "You are executing one DRIFT GraphRAG follow-up action.\n"
            "Answer the action query using only the supplied Local Search evidence.\n"
            "Score how useful this answer is for resolving the original global query.\n"
            "Return ONLY a JSON object with keys response, score, follow_up_queries. "
            "score must be an integer from 0 to 100. "
            f"follow_up_queries must contain at most {self.max_followups} short strings.\n\n"
            f"Original global query:\n{global_query}\n\n"
            f"Action query:\n{action_query}\n\n"
            f"Local evidence:\n{_format_evidence(hits)}\n"
        )
        raw = self.generator.generate(prompt).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            best = max(hits, key=lambda hit: hit.score)
            return DriftActionResult(
                response=best.text.strip(),
                score=max(0, min(100, round(best.score * 100))),
                follow_up_queries=[],
            )

        if not isinstance(payload, dict):
            payload = {}

        response = payload.get("response")
        if not isinstance(response, str) or not response.strip():
            response = max(hits, key=lambda hit: hit.score).text

        score = payload.get("score", 0)
        if not isinstance(score, int):
            try:
                score = int(score)
            except (TypeError, ValueError):
                score = 0
        score = max(0, min(100, score))

        raw_followups = payload.get("follow_up_queries", [])
        followups: list[str] = []
        if isinstance(raw_followups, list):
            for item in raw_followups:
                if isinstance(item, str):
                    item = item.strip()
                    if item and item not in followups:
                        followups.append(item)
                if len(followups) >= self.max_followups:
                    break

        return DriftActionResult(
            response=response.strip(),
            score=score,
            follow_up_queries=followups,
        )

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
            "The evidence may include a broad community-report primer plus Local Search results.\n"
            "Resolve overlaps, avoid duplication, and preserve uncertainty where evidence is incomplete.\n"
            "Do not introduce facts absent from the evidence.\n\n"
            f"Original query:\n{query}\n\n"
            f"Accumulated evidence:\n{_format_evidence(evidence)}\n"
        )
        return self.generator.generate(prompt).strip()
