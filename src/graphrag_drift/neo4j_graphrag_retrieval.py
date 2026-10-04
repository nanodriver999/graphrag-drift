from __future__ import annotations

from dataclasses import dataclass
import re

import neo4j
from neo4j import GraphDatabase, Driver
from neo4j_graphrag.retrievers import HybridCypherRetriever, VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from .config import Neo4jSettings
from .embedding import hash_embedding
from .models import CommunityReport, SearchHit
from .neo4j_retrieval import COMMUNITY_REPORT_QUERY


ENTITY_VECTOR_INDEX = "entity_embedding"
ENTITY_FULLTEXT_INDEX = "entity_fulltext"
CHUNK_VECTOR_INDEX = "chunk_embedding"
CHUNK_FULLTEXT_INDEX = "chunk_fulltext"

ENTITY_RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[r]-(neighbor:Entity)
WITH node, score,
     collect(DISTINCT coalesce(neighbor.name, neighbor.id))[0..5] AS neighbors,
     collect(DISTINCT type(r))[0..5] AS rel_types
OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(node)
WITH node, score, neighbors, rel_types,
     collect(DISTINCT coalesce(chunk.text, chunk.content))[0..3] AS chunks
RETURN 'entity:' + coalesce(node.id, elementId(node)) AS id,
       trim(
         coalesce(node.name, '') + ' | ' +
         coalesce(node.description, '') +
         CASE WHEN size(neighbors) > 0 THEN ' | neighbors: ' + reduce(s = '', x IN neighbors | s + CASE WHEN s = '' THEN '' ELSE ', ' END + coalesce(toString(x), '')) ELSE '' END +
         CASE WHEN size(rel_types) > 0 THEN ' | relationships: ' + reduce(s = '', x IN rel_types | s + CASE WHEN s = '' THEN '' ELSE ', ' END + x) ELSE '' END +
         CASE WHEN size(chunks) > 0 THEN ' | chunks: ' + reduce(s = '', x IN chunks | s + CASE WHEN s = '' THEN '' ELSE ' || ' END + coalesce(toString(x), '')) ELSE '' END
       ) AS text,
       score
"""

CHUNK_RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[:MENTIONS]->(entity:Entity)
WITH node, score,
     collect(DISTINCT coalesce(entity.name, entity.id))[0..8] AS entities
RETURN 'chunk:' + coalesce(node.id, elementId(node)) AS id,
       trim(
         coalesce(node.text, node.content, '') +
         CASE WHEN size(entities) > 0 THEN ' | mentioned entities: ' + reduce(s = '', x IN entities | s + CASE WHEN s = '' THEN '' ELSE ', ' END + coalesce(toString(x), '')) ELSE '' END
       ) AS text,
       score
"""


def _formatter(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=str(record.get("text") or ""),
        metadata={
            "id": str(record.get("id") or ""),
            "score": float(record.get("score") or 0.0),
        },
    )


def _query_terms(query: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9가-힣_]+", query)
        if len(token.strip()) >= 2
    ]


def _rerank_hits(query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
    """Merge entity/chunk results with a small exact-term bonus.

    Hybrid retrieval remains the primary signal. The lexical bonus helps Korean
    legal terms survive cross-index merging when an exact statute phrase appears
    in a Chunk but the hash embedding score is weak.
    """
    terms = _query_terms(query)
    unique: dict[str, SearchHit] = {}
    for hit in hits:
        previous = unique.get(hit.id)
        if previous is None or hit.score > previous.score:
            unique[hit.id] = hit

    def rank(hit: SearchHit) -> tuple[float, int, float]:
        lowered = hit.text.lower()
        overlap = sum(1 for term in terms if term in lowered)
        chunk_bonus = 0.05 if hit.id.startswith("chunk:") and overlap else 0.0
        blended = hit.score + min(overlap, 4) * 0.15 + chunk_bonus
        return (blended, overlap, hit.score)

    return sorted(unique.values(), key=rank, reverse=True)[:top_k]


@dataclass
class Neo4jGraphRAGRetriever:
    driver: Driver
    database: str | None = None
    use_hybrid: bool = True
    include_chunks: bool = True

    @classmethod
    def from_settings(
        cls,
        settings: Neo4jSettings,
        *,
        use_hybrid: bool = True,
        include_chunks: bool = True,
    ) -> "Neo4jGraphRAGRetriever":
        driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
        )
        driver.verify_connectivity()
        return cls(
            driver=driver,
            database=settings.database,
            use_hybrid=use_hybrid,
            include_chunks=include_chunks,
        )

    def close(self) -> None:
        self.driver.close()

    def _retriever(
        self,
        *,
        vector_index_name: str,
        fulltext_index_name: str,
        retrieval_query: str,
    ):
        if self.use_hybrid:
            return HybridCypherRetriever(
                driver=self.driver,
                vector_index_name=vector_index_name,
                fulltext_index_name=fulltext_index_name,
                retrieval_query=retrieval_query,
                result_formatter=_formatter,
                neo4j_database=self.database,
            )
        return VectorCypherRetriever(
            driver=self.driver,
            index_name=vector_index_name,
            retrieval_query=retrieval_query,
            result_formatter=_formatter,
            neo4j_database=self.database,
        )

    def _search_index(
        self,
        query: str,
        *,
        vector: list[float],
        vector_index_name: str,
        fulltext_index_name: str,
        retrieval_query: str,
        top_k: int,
    ) -> list[SearchHit]:
        retriever = self._retriever(
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            retrieval_query=retrieval_query,
        )
        if self.use_hybrid:
            result = retriever.search(
                query_text=query,
                query_vector=vector,
                top_k=top_k,
            )
        else:
            result = retriever.search(query_vector=vector, top_k=top_k)

        return [
            SearchHit(
                id=str(item.metadata.get("id", "")),
                text=item.content,
                score=float(item.metadata.get("score", 0.0)),
            )
            for item in result.items
        ]

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        vector = hash_embedding(query)
        candidate_k = max(top_k, 3)

        hits = self._search_index(
            query,
            vector=vector,
            vector_index_name=ENTITY_VECTOR_INDEX,
            fulltext_index_name=ENTITY_FULLTEXT_INDEX,
            retrieval_query=ENTITY_RETRIEVAL_QUERY,
            top_k=candidate_k,
        )

        if self.include_chunks:
            hits.extend(
                self._search_index(
                    query,
                    vector=vector,
                    vector_index_name=CHUNK_VECTOR_INDEX,
                    fulltext_index_name=CHUNK_FULLTEXT_INDEX,
                    retrieval_query=CHUNK_RETRIEVAL_QUERY,
                    top_k=candidate_k,
                )
            )

        return _rerank_hits(query, hits, top_k=top_k)

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        records, _, _ = self.driver.execute_query(
            COMMUNITY_REPORT_QUERY,
            parameters_={"query": query, "top_k": top_k},
            database_=self.database,
        )
        return [
            CommunityReport(
                id=str(record["id"]),
                summary=str(record["summary"]),
                score=float(record["score"] or 0.0),
            )
            for record in records
        ]
