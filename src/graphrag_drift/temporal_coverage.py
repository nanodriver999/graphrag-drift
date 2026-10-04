from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable


@dataclass(frozen=True)
class LawVersion:
    law_id: str
    title: str
    effective_date: str
    version_key: str = ""


@dataclass(frozen=True)
class LawCoverage:
    law_id: str
    as_of_date: str
    current_versions: tuple[LawVersion, ...]
    future_versions: tuple[LawVersion, ...]

    @property
    def has_current_version(self) -> bool:
        return bool(self.current_versions)

    @property
    def latest_current(self) -> LawVersion | None:
        if not self.current_versions:
            return None
        return max(self.current_versions, key=lambda item: item.effective_date)


def normalize_date(value: str | None = None) -> str:
    raw = value or date.today().isoformat()
    normalized = raw.replace("-", "")
    if not re.fullmatch(r"\d{8}", normalized):
        raise ValueError("date must use YYYY-MM-DD or YYYYMMDD")
    return normalized


def evaluate_law_coverage(
    law_id: str,
    versions: Iterable[LawVersion],
    *,
    as_of_date: str,
) -> LawCoverage:
    target = normalize_date(as_of_date)
    matching = [item for item in versions if item.law_id == law_id]
    current = tuple(
        sorted(
            (item for item in matching if item.effective_date and item.effective_date <= target),
            key=lambda item: item.effective_date,
        )
    )
    future = tuple(
        sorted(
            (item for item in matching if item.effective_date and item.effective_date > target),
            key=lambda item: item.effective_date,
        )
    )
    return LawCoverage(
        law_id=law_id,
        as_of_date=target,
        current_versions=current,
        future_versions=future,
    )
