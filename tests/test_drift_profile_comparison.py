from graphrag_drift.drift_profile_comparison import (
    compare_drift_profiles,
    render_drift_profile_comparison_markdown,
)


def suite(
    *,
    wall_ms: float,
    llm_calls: float,
    coverage: float,
    insufficient: float,
    evidence: float,
    actions: float,
    epochs: float,
) -> dict:
    return {
        "summary": {
            "case_count": 2,
            "modes": {
                "drift": {
                    "wall_ms": {"mean": wall_ms},
                    "llm_calls": {"mean": llm_calls},
                    "topic_coverage": {"mean": coverage},
                    "evidence_count": {"mean": evidence},
                    "action_count": {"mean": actions},
                    "epochs": {"mean": epochs},
                    "insufficient_answer_rate": insufficient,
                }
            },
        }
    }


def test_compare_drift_profiles_computes_deltas_and_ratios() -> None:
    result = compare_drift_profiles(
        {
            "economy": suite(
                wall_ms=100,
                llm_calls=4,
                coverage=0.5,
                insufficient=0.5,
                evidence=4,
                actions=2,
                epochs=1,
            ),
            "microsoft-like": suite(
                wall_ms=180,
                llm_calls=8,
                coverage=0.75,
                insufficient=0.0,
                evidence=8,
                actions=6,
                epochs=3,
            ),
        }
    )

    rows = {row["profile"]: row for row in result["profiles"]}
    expanded = rows["microsoft-like"]

    assert expanded["metrics"]["wall_ms"]["delta"] == 80.0
    assert expanded["metrics"]["llm_calls"]["ratio"] == 2.0
    assert expanded["metrics"]["topic_coverage"]["delta"] == 0.25
    assert expanded["insufficient_answer_rate_delta"] == -0.5


def test_render_profile_comparison_markdown() -> None:
    result = compare_drift_profiles(
        {
            "economy": suite(
                wall_ms=100,
                llm_calls=4,
                coverage=0.5,
                insufficient=0.5,
                evidence=4,
                actions=2,
                epochs=1,
            ),
            "microsoft-like": suite(
                wall_ms=180,
                llm_calls=8,
                coverage=0.75,
                insufficient=0.0,
                evidence=8,
                actions=6,
                epochs=3,
            ),
        }
    )

    rendered = render_drift_profile_comparison_markdown(result)

    assert "# DRIFT Profile Comparison" in rendered
    assert "| economy | 100.0 | 4.0 | 0.5 | 0.5 | 4.0 | 2.0 | 1.0 |" in rendered
    assert "| microsoft-like | 180.0 | 8.0 | 0.75 | 0.0 | 8.0 | 6.0 | 3.0 |" in rendered
    assert "topic_coverage: delta=0.25" in rendered
