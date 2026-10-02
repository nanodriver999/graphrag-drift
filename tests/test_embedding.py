from graphrag_drift.embedding import hash_embedding


def test_hash_embedding_is_deterministic_and_normalized():
    first = hash_embedding("GraphRAG local search")
    second = hash_embedding("GraphRAG local search")

    assert first == second
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9


def test_hash_embedding_changes_with_text():
    assert hash_embedding("local search") != hash_embedding("global search")
