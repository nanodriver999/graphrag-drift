from __future__ import annotations

from pathlib import Path

from graphrag_drift.profile_progress import ProfileBenchmarkProgress


def test_profile_progress_persists_results_atomically(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    progress = ProfileBenchmarkProgress.load(path)

    progress.put(
        "run:economy:abc",
        {
            "format": "graphrag-drift-mode-benchmark-v1",
            "results": [{"mode": "drift", "wall_ms": 123.0, "llm_calls": 4}],
        },
    )

    loaded = ProfileBenchmarkProgress.load(path)
    cached = loaded.get("run:economy:abc")

    assert cached is not None
    assert cached["results"][0]["wall_ms"] == 123.0
    assert cached["results"][0]["llm_calls"] == 4
    assert not path.with_suffix(".json.tmp").exists()


def test_profile_progress_returns_none_for_unknown_thread(tmp_path: Path) -> None:
    progress = ProfileBenchmarkProgress.load(tmp_path / "progress.json")

    assert progress.get("missing") is None


def test_profile_progress_reuses_matching_benchmark_identity(tmp_path: Path) -> None:
    path = tmp_path / "tracked.json"
    identity = {
        "model": "opencode/model-a",
        "snapshot_sha256": "a" * 64,
        "as_of_date": "2026-10-05",
        "app_revision": "app-sha-1",
    }
    progress = ProfileBenchmarkProgress.load(path, identity=identity)
    progress.put("run:economy:a", {"results": [{"llm_calls": 7}]})

    loaded = ProfileBenchmarkProgress.load(path, identity=identity)
    assert loaded.identity == identity
    assert loaded.get("run:economy:a")["results"][0]["llm_calls"] == 7


def test_profile_progress_rejects_changed_provenance(tmp_path: Path) -> None:
    import pytest

    path = tmp_path / "tracked.json"
    identity = {
        "model": "opencode/model-a",
        "snapshot_sha256": "a" * 64,
        "as_of_date": "2026-10-05",
        "app_revision": "app-sha-1",
    }
    progress = ProfileBenchmarkProgress.load(path, identity=identity)
    progress.put("run:economy:a", {"results": [{"llm_calls": 7}]})

    for field, value in (
        ("model", "opencode/model-b"),
        ("snapshot_sha256", "b" * 64),
        ("as_of_date", "2026-11-11"),
        ("app_revision", "app-sha-2"),
    ):
        changed = {**identity, field: value}
        with pytest.raises(ValueError, match="identity mismatch"):
            ProfileBenchmarkProgress.load(path, identity=changed)

    with pytest.raises(ValueError, match="identity mismatch"):
        ProfileBenchmarkProgress.load(path)


def test_profile_progress_rejects_legacy_cache_in_pinned_run(tmp_path: Path) -> None:
    import pytest

    path = tmp_path / "legacy.json"
    legacy = ProfileBenchmarkProgress.load(path)
    legacy.put("legacy:economy:one", {"results": [{"answer": "old"}]})

    with pytest.raises(ValueError, match="identity mismatch"):
        ProfileBenchmarkProgress.load(
            path,
            identity={"model": "opencode/model-a", "snapshot_sha256": "a" * 64},
        )
    # The historical file remains readable when no provenance is required.
    assert ProfileBenchmarkProgress.load(path).get("legacy:economy:one") is not None
