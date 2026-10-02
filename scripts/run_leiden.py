from __future__ import annotations

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from graphrag_drift.config import Neo4jSettings


def main() -> None:
    settings = Neo4jSettings.from_env()
    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        try:
            records, _, _ = driver.execute_query(
                "SHOW PROCEDURES YIELD name WHERE name = 'gds.leiden.write' RETURN name",
                database_=settings.database,
            )
        except ClientError as exc:
            print(f"Leiden unavailable: unable to inspect GDS procedures ({exc.code})")
            return

        if not records:
            print("Leiden unavailable: gds.leiden.write is not exposed on this AuraDB instance")
            return

        # Aura Graph Analytics (AGA) requires a separate GDS Session. Creating one
        # can incur additional cost, so this repository does not auto-provision it.
        # We intentionally probe the projection API without session creation and
        # classify the expected session/memory error as a safe skip.
        try:
            driver.execute_query(
                """
                CALL gds.graph.project(
                  'graphrag_leiden_probe',
                  'Entity',
                  'USES',
                  {}
                )
                YIELD graphName
                RETURN graphName
                """,
                database_=settings.database,
            )
        except ClientError as exc:
            message = str(exc)
            if "sessionId" in message or "memory" in message or "session creation" in message:
                print(
                    "Leiden skipped: this AuraDB uses Aura Graph Analytics sessions. "
                    "A separate billed GDS session must be explicitly provisioned before "
                    "running Leiden."
                )
                return
            print(f"Leiden unavailable: projection probe failed ({exc.code})")
            return

        # If the projection unexpectedly succeeds on an attached/self-managed GDS
        # environment, clean up the probe and avoid mutating application data here.
        try:
            driver.execute_query(
                "CALL gds.graph.drop('graphrag_leiden_probe', false)",
                database_=settings.database,
            )
        except ClientError:
            pass

        print(
            "Leiden available on the attached GDS runtime; automatic execution is "
            "disabled in this smoke test to avoid unexpected graph mutations."
        )


if __name__ == "__main__":
    main()
