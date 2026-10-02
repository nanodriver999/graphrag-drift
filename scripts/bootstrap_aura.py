from __future__ import annotations

from neo4j import GraphDatabase

from graphrag_drift.config import Neo4jSettings


BOOTSTRAP_QUERY = """
MERGE (a:Entity {id: 'graphrag'})
SET a.name = 'GraphRAG',
    a.description = 'GraphRAG combines graph structure with retrieval augmented generation'

MERGE (b:Entity {id: 'drift'})
SET b.name = 'DRIFT Search',
    b.description = 'DRIFT iteratively explores local graph context using follow-up questions'

MERGE (c:Entity {id: 'global'})
SET c.name = 'Global Search',
    c.description = 'Global search aggregates community reports using map and reduce'

MERGE (a)-[:USES]->(c)
MERGE (a)-[:USES]->(b)

MERGE (chunk1:Chunk {id: 'chunk-local'})
SET chunk1.text = 'Local search starts from relevant entities and expands graph context'
MERGE (chunk1)-[:MENTIONS]->(a)

MERGE (community:Community {id: 'community-search'})
SET community.name = 'GraphRAG Search Strategies',
    community.summary = 'Community covering Local, Global and DRIFT GraphRAG search strategies',
    community.rank = 1.0

MERGE (report:CommunityReport {id: 'report-search'})
SET report.summary = 'GraphRAG supports local entity-centered retrieval, global community synthesis, and DRIFT iterative exploration',
    report.rank = 1.0

MERGE (community)-[:HAS_REPORT]->(report)
MERGE (a)-[:IN_COMMUNITY]->(community)
MERGE (b)-[:IN_COMMUNITY]->(community)
MERGE (c)-[:IN_COMMUNITY]->(community)

RETURN count(*) AS touched
"""


def main() -> None:
    settings = Neo4jSettings.from_env()
    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            BOOTSTRAP_QUERY,
            database_=settings.database,
        )
        print(f"Aura bootstrap complete: {records[0]['touched']} result row")


if __name__ == "__main__":
    main()
