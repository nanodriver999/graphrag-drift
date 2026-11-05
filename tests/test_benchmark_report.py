from graphrag_drift.benchmark_report import render_benchmark_markdown


def test_render_benchmark_markdown_includes_summary_and_answers() -> None:
    result = {
        "summary": {
            "case_count": 1,
            "modes": {
                "local": {
                    "wall_ms": {"mean": 10.0},
                    "llm_calls": {"mean": 1.0},
                    "evidence_count": {"mean": 2.0},
                    "topic_coverage": {"mean": 0.5},
                    "insufficient_answer_rate": 0.0,
                },
                "global": {
                    "wall_ms": {"mean": 20.0},
                    "llm_calls": {"mean": 3.0},
                    "evidence_count": {"mean": 3.0},
                    "topic_coverage": {"mean": 0.75},
                    "insufficient_answer_rate": 0.0,
                },
                "drift": {
                    "wall_ms": {"mean": 30.0},
                    "llm_calls": {"mean": 4.0},
                    "evidence_count": {"mean": 5.0},
                    "topic_coverage": {"mean": 1.0},
                    "insufficient_answer_rate": 0.0,
                    "action_count": {"mean": 3.0},
                    "epochs": {"mean": 2.0},
                },
            },
        },
        "cases": [
            {
                "id": "q1",
                "query": "question",
                "benchmark": {
                    "results": [
                        {
                            "mode": "local",
                            "answer": "answer text",
                            "topic_coverage": 0.5,
                            "insufficient_answer": False,
                        }
                    ]
                },
            }
        ],
    }

    rendered = render_benchmark_markdown(result)

    assert "# GraphRAG Benchmark Summary" in rendered
    assert "| drift | 30.0 | 4.0 | 5.0 | 1.0 | 0.0 | - | 3.0 | 2.0 |" in rendered
    assert "### q1" in rendered
    assert "answer text" in rendered


def test_render_benchmark_markdown_includes_drift_settings() -> None:
    result = {
        "summary": {"case_count": 1, "modes": {}},
        "cases": [
            {
                "id": "case-1",
                "query": "q",
                "benchmark": {
                    "settings": {
                        "local_top_k": 4,
                        "local_community_prop": 0.25,
                        "local_text_unit_prop": 0.50,
                        "global_top_k": 3,
                        "global_map_concurrency": 8,
                        "drift_top_k": 2,
                        "drift_profile": "microsoft-like",
                        "drift_max_followups": 20,
                        "drift_max_depth": 3,
                        "drift_max_actions_per_depth": 4,
                        "drift_community_top_k": 1,
                        "drift_action_concurrency": 8,
                        "drift_primer_folds": 5,
                        "drift_primer_concurrency": 5,
                    },
                    "results": [],
                },
            }
        ],
    }

    markdown = render_benchmark_markdown(result)

    assert "Local community proportion: 0.25" in markdown
    assert "Local text-unit proportion: 0.5" in markdown
    assert "Global map concurrency: 8" in markdown
    assert "DRIFT profile: microsoft-like" in markdown
    assert "DRIFT max follow-ups/action: 20" in markdown
    assert "DRIFT max depth: 3" in markdown
    assert "DRIFT action concurrency: 8" in markdown
    assert "DRIFT primer folds: 5" in markdown
