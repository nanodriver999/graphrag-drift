from graphrag_drift.embedding import hash_embedding


def test_hash_embedding_is_deterministic_and_normalized():
    first = hash_embedding("GraphRAG local search")
    second = hash_embedding("GraphRAG local search")

    assert first == second
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9


def test_hash_embedding_changes_with_text():
    assert hash_embedding("local search") != hash_embedding("global search")


def test_hash_embedding_never_returns_zero_vector_for_nonempty_text():
    vector = hash_embedding("방사선안전관리자 업무")

    assert any(value != 0.0 for value in vector)
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-9
