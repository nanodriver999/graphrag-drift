from __future__ import annotations

from dataclasses import dataclass
import re

from neo4j import GraphDatabase, Driver

from .config import Neo4jSettings
from .embedding import hash_embedding
from .models import CommunityReport, SearchHit
from .neo4j_retrieval import COMMUNITY_REPORT_QUERY


ENTITY_VECTOR_INDEX = "entity_embedding"
ENTITY_FULLTEXT_INDEX = "entity_fulltext"
CHUNK_VECTOR_INDEX = "chunk_embedding"
CHUNK_FULLTEXT_INDEX = "chunk_fulltext"

ENTITY_HYBRID_QUERY = """
CALL () {
  CALL db.index.vector.queryNodes($vector_index, $candidate_k, $query_vector)
  YIELD node, score
  WITH collect({node: node, score: score}) AS rows, max(score) AS max_score
  UNWIND rows AS row
  RETURN row.node AS node,
         CASE WHEN max_score IS NULL OR max_score = 0 THEN 0.0 ELSE row.score / max_score END AS score
  UNION
  CALL db.index.fulltext.queryNodes($fulltext_index, $query_text, {limit: $candidate_k})
  YIELD node, score
  WITH collect({node: node, score: score}) AS rows, max(score) AS max_score
  UNWIND rows AS row
  RETURN row.node AS node,
         CASE WHEN max_score IS NULL OR max_score = 0 THEN 0.0 ELSE row.score / max_score END AS score
}
WITH node, max(score) AS score
ORDER BY score DESC, elementId(node)
LIMIT $candidate_k
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

CHUNK_HYBRID_QUERY = """
CALL () {
  CALL db.index.vector.queryNodes($vector_index, $candidate_k, $query_vector)
  YIELD node, score
  WITH collect({node: node, score: score}) AS rows, max(score) AS max_score
  UNWIND rows AS row
  RETURN row.node AS node,
         CASE WHEN max_score IS NULL OR max_score = 0 THEN 0.0 ELSE row.score / max_score END AS score
  UNION
  CALL db.index.fulltext.queryNodes($fulltext_index, $query_text, {limit: $candidate_k})
  YIELD node, score
  WITH collect({node: node, score: score}) AS rows, max(score) AS max_score
  UNWIND rows AS row
  RETURN row.node AS node,
         CASE WHEN max_score IS NULL OR max_score = 0 THEN 0.0 ELSE row.score / max_score END AS score
}
WITH node, max(score) AS score
ORDER BY score DESC, elementId(node)
LIMIT $candidate_k
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

ENTITY_VECTOR_QUERY = """
CALL db.index.vector.queryNodes($vector_index, $candidate_k, $query_vector)
YIELD node, score
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
ORDER BY score DESC
"""

CHUNK_VECTOR_QUERY = """
CALL db.index.vector.queryNodes($vector_index, $candidate_k, $query_vector)
YIELD node, score
OPTIONAL MATCH (node)-[:MENTIONS]->(entity:Entity)
WITH node, score,
     collect(DISTINCT coalesce(entity.name, entity.id))[0..8] AS entities
RETURN 'chunk:' + coalesce(node.id, elementId(node)) AS id,
       trim(
         coalesce(node.text, node.content, '') +
         CASE WHEN size(entities) > 0 THEN ' | mentioned entities: ' + reduce(s = '', x IN entities | s + CASE WHEN s = '' THEN '' ELSE ', ' END + coalesce(toString(x), '')) ELSE '' END
       ) AS text,
       score
ORDER BY score DESC
"""


def _query_terms(query: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9가-힣_]+", query)
        if len(token.strip()) >= 2
    ]


def _rerank_hits(query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
    """Merge entity/chunk results with a small exact-term bonus."""
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

    def _run_search(
        self,
        cypher: str,
        *,
        vector_index: str,
        fulltext_index: str,
        query: str,
        vector: list[float],
        candidate_k: int,
    ) -> list[SearchHit]:
        params = {
            "vector_index": vector_index,
            "query_vector": vector,
            "candidate_k": candidate_k,
        }
        if self.use_hybrid:
            params["fulltext_index"] = fulltext_index
            params["query_text"] = query

        records, _, _ = self.driver.execute_query(
            cypher,
            parameters_=params,
            database_=self.database,
        )
        return [
            SearchHit(
                id=str(record["id"]),
                text=str(record["text"] or ""),
                score=float(record["score"] or 0.0),
            )
            for record in records
        ]

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        vector = hash_embedding(query)
        candidate_k = max(top_k * 2, 6)

        entity_query = ENTITY_HYBRID_QUERY if self.use_hybrid else ENTITY_VECTOR_QUERY
        hits = self._run_search(
            entity_query,
            vector_index=ENTITY_VECTOR_INDEX,
            fulltext_index=ENTITY_FULLTEXT_INDEX,
            query=query,
            vector=vector,
            candidate_k=candidate_k,
        )

        if self.include_chunks:
            chunk_query = CHUNK_HYBRID_QUERY if self.use_hybrid else CHUNK_VECTOR_QUERY
            hits.extend(
                self._run_search(
                    chunk_query,
                    vector_index=CHUNK_VECTOR_INDEX,
                    fulltext_index=CHUNK_FULLTEXT_INDEX,
                    query=query,
                    vector=vector,
                    candidate_k=candidate_k,
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
