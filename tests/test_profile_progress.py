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
