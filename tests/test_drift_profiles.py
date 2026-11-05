from __future__ import annotations

import pytest

from graphrag_drift.drift_profiles import (
    get_drift_search_profile,
    resolve_drift_search_profile,
)


def test_economy_profile_preserves_existing_defaults() -> None:
    profile = get_drift_search_profile("economy")

    assert profile.max_depth == 2
    assert profile.max_actions_per_depth == 2
    assert profile.community_top_k == 1
    assert profile.action_concurrency == 4
    assert profile.primer_folds == 5
    assert profile.primer_concurrency == 5
    assert profile.max_followups == 2


def test_microsoft_like_profile_is_broader_than_economy() -> None:
    economy = get_drift_search_profile("economy")
    profile = get_drift_search_profile("microsoft-like")

    assert profile.max_depth == 3
    assert profile.max_actions_per_depth > economy.max_actions_per_depth
    assert profile.action_concurrency > economy.action_concurrency
    assert profile.max_followups > economy.max_followups


def test_profile_overrides_do_not_mutate_named_profile() -> None:
    resolved = resolve_drift_search_profile(
        "microsoft-like",
        max_actions_per_depth=4,
        max_followups=6,
    )

    assert resolved.name == "microsoft-like"
    assert resolved.max_actions_per_depth == 4
    assert resolved.max_followups == 6
    assert get_drift_search_profile("microsoft-like").max_actions_per_depth == 8


def test_profile_engine_kwargs_exclude_reasoner_followup_setting() -> None:
    profile = get_drift_search_profile("economy")

    kwargs = profile.engine_kwargs()

    assert kwargs["max_depth"] == 2
    assert "max_followups" not in kwargs


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown DRIFT profile"):
        get_drift_search_profile("unknown")


def test_invalid_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        resolve_drift_search_profile("economy", max_depth=0)
