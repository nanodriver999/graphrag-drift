from __future__ import annotations

from dataclasses import dataclass

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

RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[r]-(neighbor:Entity)
WITH node, score,
     collect(DISTINCT coalesce(neighbor.name, neighbor.id))[0..5] AS neighbors,
     collect(DISTINCT type(r))[0..5] AS rel_types
OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(node)
WITH node, score, neighbors, rel_types,
     collect(DISTINCT coalesce(chunk.text, chunk.content))[0..3] AS chunks
RETURN coalesce(node.id, elementId(node)) AS id,
       trim(
         coalesce(node.name, '') + ' | ' +
         coalesce(node.description, '') +
         CASE WHEN size(neighbors) > 0 THEN ' | neighbors: ' + reduce(s = '', x IN neighbors | s + CASE WHEN s = '' THEN '' ELSE ', ' END + coalesce(toString(x), '')) ELSE '' END +
         CASE WHEN size(rel_types) > 0 THEN ' | relationships: ' + reduce(s = '', x IN rel_types | s + CASE WHEN s = '' THEN '' ELSE ', ' END + x) ELSE '' END +
         CASE WHEN size(chunks) > 0 THEN ' | chunks: ' + reduce(s = '', x IN chunks | s + CASE WHEN s = '' THEN '' ELSE ' || ' END + coalesce(toString(x), '')) ELSE '' END
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


@dataclass
class Neo4jGraphRAGRetriever:
    driver: Driver
    database: str | None = None
    use_hybrid: bool = True

    @classmethod
    def from_settings(
        cls,
        settings: Neo4jSettings,
        *,
        use_hybrid: bool = True,
    ) -> "Neo4jGraphRAGRetriever":
        driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
        )
        driver.verify_connectivity()
        return cls(driver=driver, database=settings.database, use_hybrid=use_hybrid)

    def close(self) -> None:
        self.driver.close()

    def _retriever(self):
        if self.use_hybrid:
            return HybridCypherRetriever(
                driver=self.driver,
                vector_index_name=ENTITY_VECTOR_INDEX,
                fulltext_index_name=ENTITY_FULLTEXT_INDEX,
                retrieval_query=RETRIEVAL_QUERY,
                result_formatter=_formatter,
                neo4j_database=self.database,
            )
        return VectorCypherRetriever(
            driver=self.driver,
            index_name=ENTITY_VECTOR_INDEX,
            retrieval_query=RETRIEVAL_QUERY,
            result_formatter=_formatter,
            neo4j_database=self.database,
        )

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        vector = hash_embedding(query)
        retriever = self._retriever()
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
