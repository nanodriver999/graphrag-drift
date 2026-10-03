from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class CompatibleHandler(BaseHTTPRequestHandler):
    prompts: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        assert self.path == "/v1/chat/completions"
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        prompt = payload["messages"][0]["content"]
        type(self).prompts.append(prompt)

        if "community analyst" in prompt:
            content = "mapped safety duty"
        elif "global answer synthesizer" in prompt:
            content = "final grounded answer"
        else:
            content = "unexpected prompt"

        body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_llm_cached_global_search_cli_runs_map_reduce_against_compatible_endpoint(
    tmp_path: Path,
) -> None:
    dump_sha = "d" * 64
    prefix = dump_sha[:12]
    report_dir = tmp_path / "community_reports" / prefix
    report_file = report_dir / "community-leiden-8.json"

    _write_json(
        tmp_path / "data" / "manifest.json",
        {"dump": {"zip_sha256": dump_sha}},
    )
    _write_json(
        report_file,
        {
            "community_id": "leiden-8",
            "report": {
                "title": "Radiation safety",
                "summary": "Licensed users appoint a radiation safety manager.",
                "key_findings": ["Registered agents may perform delegated work."],
            },
        },
    )
    _write_json(
        report_dir / "manifest.json",
        {
            "reports": [
                {
                    "community_id": "leiden-8",
                    "file": str(report_file.relative_to(tmp_path)),
                }
            ]
        },
    )

    CompatibleHandler.prompts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), CompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        env = os.environ.copy()
        env["OPENAI_COMPATIBLE_BASE_URL"] = (
            f"http://127.0.0.1:{server.server_address[1]}"
        )
        env["OPENAI_COMPATIBLE_MODEL"] = "test-model"
        env.pop("OPENAI_COMPATIBLE_API_KEY", None)

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_llm_cached_global_search.py",
                "방사선안전관리자 의무",
                "--data-repo",
                str(tmp_path),
                "--top-k",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    result = json.loads(completed.stdout)
    assert result["partials"] == [["mapped safety duty", 1.0]]
    assert result["answer"] == "final grounded answer"

    assert len(CompatibleHandler.prompts) == 2
    assert "community analyst" in CompatibleHandler.prompts[0]
    assert "leiden-8" in CompatibleHandler.prompts[0]
    assert "global answer synthesizer" in CompatibleHandler.prompts[1]
    assert "mapped safety duty" in CompatibleHandler.prompts[1]
