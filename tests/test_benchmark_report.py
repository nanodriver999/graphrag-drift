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
    assert "| drift | 30.0 | 4.0 | 5.0 | 1.0 | 0.0 | 3.0 | 2.0 |" in rendered
    assert "### q1" in rendered
    assert "answer text" in rendered
