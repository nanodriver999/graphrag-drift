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

    @classmethod
    def load(cls, path: str | Path) -> "ProfileBenchmarkProgress":
        target = Path(path)
        if not target.exists():
            return cls(target)

        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise ValueError(f"Unsupported profile progress file: {target}")

        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            raise ValueError("Profile progress entries must be an object")
        return cls(
            target,
            {
                str(key): value
                for key, value in entries.items()
                if isinstance(value, dict)
            },
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
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)
