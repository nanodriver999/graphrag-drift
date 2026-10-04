from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .global_llm_reasoner import TextGenerator

REPORT_FORMAT = "graphrag-drift-community-report-v1"
MANIFEST_FORMAT = "graphrag-drift-community-report-manifest-v1"
HASH_METHOD = 'sha256(dump_zip_sha256 + "\\n" + context_sha256 + "\\n" + report_format)'


def report_content_hash(dump_sha: str, context_sha: str) -> str:
    material = f"{dump_sha}\n{context_sha}\n{REPORT_FORMAT}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Community report generator must return a JSON object")
    return payload


def _validate_report(body: dict[str, Any]) -> dict[str, Any]:
    title = body.get("title")
    summary = body.get("summary")
    findings = body.get("key_findings")
    entities = body.get("entities", [])
    notes = body.get("notes", "")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Generated report is missing title")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Generated report is missing summary")
    if not isinstance(findings, list) or not findings or any(
        not isinstance(item, str) or not item.strip() for item in findings
    ):
        raise ValueError("Generated report key_findings must be a non-empty string list")
    if not isinstance(entities, list) or any(not isinstance(item, str) for item in entities):
        raise ValueError("Generated report entities must be a string list")
    if not isinstance(notes, str):
        raise ValueError("Generated report notes must be a string")

    return {
        "title": title.strip(),
        "summary": summary.strip(),
        "key_findings": [item.strip() for item in findings],
        "entities": [item.strip() for item in entities if item.strip()],
        "notes": notes.strip(),
    }


@dataclass
class CommunityReportCacheBuilder:
    generator: TextGenerator
    generator_name: str = "openai-compatible"
    max_part_chars: int = 60000

    def _summarize_part(self, community_id: str, part: dict[str, Any]) -> dict[str, Any]:
        serialized = json.dumps(part, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > self.max_part_chars:
            serialized = serialized[: self.max_part_chars]

        prompt = (
            "You summarize one evidence partition for a GraphRAG community report.\n"
            "Use only the supplied evidence. Preserve legal article numbers, named entities, "
            "relationships, duties, exceptions, and effective-date qualifications when present.\n"
            "Return ONLY JSON with keys summary, key_points, entities.\n"
            "key_points and entities must be arrays of strings.\n\n"
            f"Community: {community_id}\nEvidence:\n{serialized}"
        )
        payload = _parse_json_object(self.generator.generate(prompt))
        summary = payload.get("summary")
        points = payload.get("key_points", [])
        entities = payload.get("entities", [])
        if not isinstance(summary, str):
            raise ValueError("Part summary is missing summary")
        if not isinstance(points, list) or any(not isinstance(item, str) for item in points):
            raise ValueError("Part summary key_points must be strings")
        if not isinstance(entities, list) or any(not isinstance(item, str) for item in entities):
            raise ValueError("Part summary entities must be strings")
        return {
            "summary": summary.strip(),
            "key_points": [item.strip() for item in points if item.strip()],
            "entities": [item.strip() for item in entities if item.strip()],
        }

    def _generate_report(
        self,
        community_id: str,
        overview: dict[str, Any],
        part_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        compact = {
            "community": overview.get("community"),
            "entities": overview.get("entities", []),
            "relationships": overview.get("relationships", []),
            "evidence_node_count": overview.get("evidence_node_count", 0),
            "part_summaries": part_summaries,
        }
        prompt = (
            "Create the final GraphRAG Community Report from the supplied overview and evidence summaries.\n"
            "Use only the supplied material. Do not invent legal requirements or relationships.\n"
            "Return ONLY JSON with keys title, summary, key_findings, entities, notes.\n"
            "key_findings must be a non-empty array of concise strings; entities must be an array of strings.\n\n"
            f"Community: {community_id}\nInput:\n"
            + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        )
        return _validate_report(_parse_json_object(self.generator.generate(prompt)))

    def build(self, repository_root: str | Path) -> dict[str, Any]:
        root = Path(repository_root)
        dump_manifest = _load_json(root / "data" / "manifest.json")
        input_manifest = _load_json(root / "community_report_inputs" / "manifest.json")

        dump_sha = str(dump_manifest["dump"]["zip_sha256"])
        prefix = dump_sha[:12]
        output_dir = root / "community_reports" / prefix
        output_dir.mkdir(parents=True, exist_ok=True)

        report_rows: list[dict[str, Any]] = []
        generated = 0
        reused = 0

        for row in input_manifest.get("communities", []):
            community_id = str(row["community_id"])
            context_sha = str(row["context_sha256"])
            content_hash = report_content_hash(dump_sha, context_sha)
            relative_file = Path("community_reports") / prefix / f"community-{community_id}.json"
            output_file = root / relative_file

            if output_file.exists():
                existing = _load_json(output_file)
                if existing.get("content_hash") == content_hash:
                    reused += 1
                    report_rows.append(
                        {
                            "community_id": community_id,
                            "context_sha256": context_sha,
                            "content_hash": content_hash,
                            "file": relative_file.as_posix(),
                        }
                    )
                    continue

            overview = _load_json(root / str(row["overview_file"]))
            part_summaries = []
            for part_row in row.get("parts", []):
                part = _load_json(root / str(part_row["file"]))
                part_summaries.append(self._summarize_part(community_id, part))

            report_body = self._generate_report(community_id, overview, part_summaries)
            payload = {
                "format": REPORT_FORMAT,
                "community_id": community_id,
                "source": {
                    "dump_zip_sha256": dump_sha,
                    "context_sha256": context_sha,
                    "leiden_property": "leidenCommunityId",
                    "generator": self.generator_name,
                    "content_hash_method": HASH_METHOD,
                },
                "content_hash": content_hash,
                "report": report_body,
            }
            output_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            generated += 1
            report_rows.append(
                {
                    "community_id": community_id,
                    "context_sha256": context_sha,
                    "content_hash": content_hash,
                    "file": relative_file.as_posix(),
                }
            )

        manifest = {
            "format": MANIFEST_FORMAT,
            "dump_zip_sha256": dump_sha,
            "dump_prefix": prefix,
            "generator": self.generator_name,
            "report_format": REPORT_FORMAT,
            "content_hash_method": HASH_METHOD,
            "report_count": len(report_rows),
            "reports": report_rows,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "manifest": manifest,
            "generated": generated,
            "reused": reused,
            "manifest_path": str(output_dir / "manifest.json"),
        }
