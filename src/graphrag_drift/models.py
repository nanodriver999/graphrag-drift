from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    id: str
    text: str
    score: float = 1.0


@dataclass(frozen=True)
class CommunityReport:
    id: str
    summary: str
    score: float = 1.0
    # Optional input-context lineage for cached/generated reports. This is not
    # proof that each sentence of the summary is supported by those sources.
    source_provenance: dict[str, object] | None = None


@dataclass(frozen=True)
class DriftPrimerResult:
    intermediate_answer: str
    score: int
    follow_up_queries: list[str]


@dataclass(frozen=True)
class DriftActionResult:
    response: str
    score: int
    follow_up_queries: list[str]
