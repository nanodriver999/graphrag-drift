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
