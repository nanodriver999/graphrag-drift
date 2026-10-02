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
