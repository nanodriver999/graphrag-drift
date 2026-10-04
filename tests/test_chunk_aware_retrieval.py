from graphrag_drift.models import SearchHit
from graphrag_drift.neo4j_graphrag_retrieval import _rerank_hits


def test_chunk_hit_with_exact_legal_terms_can_outrank_weaker_entity_hit() -> None:
    hits = [
        SearchHit(
            id="entity:safety-manager",
            text="방사선 안전 관련 일반 설명",
            score=0.90,
        ),
        SearchHit(
            id="chunk:article-1",
            text="방사선안전관리자는 사용 개시 전에 선임하여 신고하여야 한다.",
            score=0.55,
        ),
    ]

    ranked = _rerank_hits(
        "방사선안전관리자 선임 신고 의무",
        hits,
        top_k=2,
    )

    assert ranked[0].id == "chunk:article-1"


def test_rerank_deduplicates_ids_and_respects_top_k() -> None:
    hits = [
        SearchHit("chunk:1", "업무대행자 등록 의무", 0.4),
        SearchHit("chunk:1", "업무대행자 등록 의무", 0.8),
        SearchHit("entity:1", "업무대행자", 0.7),
    ]

    ranked = _rerank_hits("업무대행자 등록 의무", hits, top_k=1)

    assert len(ranked) == 1
    assert ranked[0].id == "chunk:1"
    assert ranked[0].score == 0.8
