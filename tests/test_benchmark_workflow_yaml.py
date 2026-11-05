from pathlib import Path

import yaml


def test_benchmark_workflow_yaml_parses() -> None:
    path = Path(".github/workflows/benchmark-suite.yml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
    assert payload["name"] == "Run GraphRAG benchmark suite"
    assert "jobs" in payload
    assert "benchmark" in payload["jobs"]
