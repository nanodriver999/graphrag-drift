from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    username: str
    password: str
    database: str | None = None

    @classmethod
    def from_env(cls) -> "Neo4jSettings":
        missing = [
            name
            for name in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
            if not os.getenv(name)
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required Neo4j environment variables: {joined}")

        return cls(
            uri=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            database=os.getenv("NEO4J_DATABASE") or None,
        )
