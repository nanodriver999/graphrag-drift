from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase, Driver

from .config import Neo4jSettings
from .models import CommunityReport, SearchHit


LOCAL_SEARCH_QUERY = """
MATCH (e:Entity)
WITH e,
     CASE
       WHEN toLower(coalesce(e.name, '')) CONTAINS toLower($query) THEN 2.0
       WHEN toLower(coalesce(e.description, '')) CONTAINS toLower($query) THEN 1.0
       ELSE 0.0
     END AS score
WHERE score > 0
OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
WITH e, score,
     collect(DISTINCT coalesce(neighbor.name, neighbor.id))[0..5] AS neighbors,
     collect(DISTINCT type(r))[0..5] AS rel_types
OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(e)
WITH e, score, neighbors, rel_types,
     collect(DISTINCT coalesce(chunk.text, chunk.content))[0..3] AS chunks
RETURN coalesce(e.id, elementId(e)) AS id,
       trim(
         coalesce(e.name, '') + ' | ' +
         coalesce(e.description, '') +
         CASE WHEN size(neighbors) > 0 THEN ' | neighbors: ' + reduce(s = '', x IN neighbors | s + CASE WHEN s = '' THEN '' ELSE ', ' END + coalesce(toString(x), '')) ELSE '' END +
         CASE WHEN size(rel_types) > 0 THEN ' | relationships: ' + reduce(s = '', x IN rel_types | s + CASE WHEN s = '' THEN '' ELSE ', ' END + x) ELSE '' END +
         CASE WHEN size(chunks) > 0 THEN ' | chunks: ' + reduce(s = '', x IN chunks | s + CASE WHEN s = '' THEN '' ELSE ' || ' END + coalesce(toString(x), '')) ELSE '' END
       ) AS text,
       score
ORDER BY score DESC
LIMIT $top_k
"""

COMMUNITY_REPORT_QUERY = """
MATCH (c:Community)
OPTIONAL MATCH (c)-[:HAS_REPORT]->(report:CommunityReport)
WITH c, report,
     CASE
       WHEN toLower(coalesce(report.summary, c.summary, '')) CONTAINS toLower($query) THEN 2.0
       WHEN toLower(coalesce(c.title, c.name, '')) CONTAINS toLower($query) THEN 1.0
       ELSE coalesce(report.rank, c.rank, 0.0)
     END AS score
RETURN coalesce(report.id, c.id, elementId(c)) AS id,
       coalesce(report.summary, c.summary, c.title, c.name, '') AS summary,
       toFloat(score) AS score
ORDER BY score DESC
LIMIT $top_k
"""


@dataclass
class Neo4jRetriever:
    driver: Driver
    database: str | None = None

    @classmethod
    def from_settings(cls, settings: Neo4jSettings) -> "Neo4jRetriever":
        driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
        )
        driver.verify_connectivity()
        return cls(driver=driver, database=settings.database)

    def close(self) -> None:
        self.driver.close()

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            cypher,
            parameters_=params,
            database_=self.database,
        )
        return [record.data() for record in records]

    def local_search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        rows = self._run(LOCAL_SEARCH_QUERY, query=query, top_k=top_k)
        return [
            SearchHit(
                id=str(row["id"]),
                text=str(row["text"]),
                score=float(row["score"] or 0.0),
            )
            for row in rows
        ]

    def community_reports(self, query: str, *, top_k: int = 5) -> list[CommunityReport]:
        rows = self._run(COMMUNITY_REPORT_QUERY, query=query, top_k=top_k)
        return [
            CommunityReport(
                id=str(row["id"]),
                summary=str(row["summary"]),
                score=float(row["score"] or 0.0),
            )
            for row in rows
        ]
