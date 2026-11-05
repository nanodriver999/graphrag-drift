from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class DriftSearchProfile:
    """Named DRIFT search-budget preset.

    Profiles describe orchestration/search breadth. They intentionally stay separate
    from provider/model settings so the same search policy can be benchmarked across
    different LLM backends.
    """

    name: str
    max_depth: int
    max_actions_per_depth: int
    community_top_k: int
    action_concurrency: int
    primer_folds: int
    primer_concurrency: int
    max_followups: int

    def __post_init__(self) -> None:
        positive = {
            "max_depth": self.max_depth,
            "max_actions_per_depth": self.max_actions_per_depth,
            "action_concurrency": self.action_concurrency,
            "primer_folds": self.primer_folds,
            "primer_concurrency": self.primer_concurrency,
            "max_followups": self.max_followups,
        }
        for field_name, value in positive.items():
            if value < 1:
                raise ValueError(f"{field_name} must be >= 1")
        if self.community_top_k < 0:
            raise ValueError("community_top_k must be >= 0")

    def engine_kwargs(self) -> dict[str, int]:
        return {
            "max_depth": self.max_depth,
            "max_actions_per_depth": self.max_actions_per_depth,
            "community_top_k": self.community_top_k,
            "action_concurrency": self.action_concurrency,
            "primer_folds": self.primer_folds,
            "primer_concurrency": self.primer_concurrency,
        }

    def serialize(self) -> dict[str, Any]:
        return asdict(self)


DRIFT_SEARCH_PROFILES: dict[str, DriftSearchProfile] = {
    # Preserve the project's existing cost-conscious defaults.
    "economy": DriftSearchProfile(
        name="economy",
        max_depth=2,
        max_actions_per_depth=2,
        community_top_k=1,
        action_concurrency=4,
        primer_folds=5,
        primer_concurrency=5,
        max_followups=2,
    ),
    # Closer to Microsoft GraphRAG DRIFT's broader/deeper exploration envelope.
    # This is intentionally named "microsoft-like", not "microsoft", because the
    # orchestration/data model in this repository is not byte-for-byte identical.
    "microsoft-like": DriftSearchProfile(
        name="microsoft-like",
        max_depth=3,
        max_actions_per_depth=8,
        community_top_k=2,
        action_concurrency=32,
        primer_folds=5,
        primer_concurrency=32,
        max_followups=20,
    ),
}


def get_drift_search_profile(name: str = "economy") -> DriftSearchProfile:
    try:
        return DRIFT_SEARCH_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DRIFT_SEARCH_PROFILES))
        raise ValueError(f"Unknown DRIFT profile {name!r}; choose one of: {choices}") from exc


def resolve_drift_search_profile(
    name: str = "economy",
    **overrides: int | None,
) -> DriftSearchProfile:
    profile = get_drift_search_profile(name)
    valid = {
        "max_depth",
        "max_actions_per_depth",
        "community_top_k",
        "action_concurrency",
        "primer_folds",
        "primer_concurrency",
        "max_followups",
    }
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown DRIFT profile overrides: {sorted(unknown)}")

    values = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }
    return replace(profile, **values)
