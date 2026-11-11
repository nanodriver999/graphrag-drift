from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FORMAT = "graphrag-drift-profile-progress-v1"


@dataclass
class ProfileBenchmarkProgress:
    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    identity: dict[str, Any] | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        identity: dict[str, Any] | None = None,
    ) -> "ProfileBenchmarkProgress":
        """Reject cached results from a different model, snapshot or app revision.

        Legacy progress without an identity remains readable only in legacy mode.
        To start a reproducible run, use a new progress path if the old file
        contains unversioned measurements.
        """
        target = Path(path)
        if not target.exists():
            return cls(target, identity=identity)

        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise ValueError(f"Unsupported profile progress file: {target}")

        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            raise ValueError("Profile progress entries must be an object")
        stored_identity = payload.get("identity")
        if stored_identity is not None and not isinstance(stored_identity, dict):
            raise ValueError("Profile progress identity must be an object")

        if stored_identity != identity:
            if entries:
                raise ValueError(
                    "Benchmark progress identity mismatch (model, snapshot, "
                    "date, or app revision). Use a new run ID/progress file "
                    "instead of combining incompatible measurements."
                )
            stored_identity = identity

        return cls(
            path=target,
            entries={
                str(key): value
                for key, value in entries.items()
                if isinstance(value, dict)
            },
            identity=stored_identity,
        )

    def get(self, thread_id: str) -> dict[str, Any] | None:
        return self.entries.get(thread_id)

    def put(self, thread_id: str, result: dict[str, Any]) -> None:
        self.entries[thread_id] = result
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT,
            "entries": self.entries,
        }
        if self.identity is not None:
            payload["identity"] = self.identity
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)
