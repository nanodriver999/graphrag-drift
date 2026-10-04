from __future__ import annotations

import importlib.util
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "opencode_openai_shim.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("opencode_openai_shim", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeOpenCodeHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def _write(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/global/health":
            self._write({"healthy": True, "version": "test"})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(
            {
                "path": self.path,
                "payload": payload,
            }
        )

        if self.path == "/session":
            self._write({"id": "ses_test"})
            return

        if self.path == "/session/ses_test/message":
            self._write(
                {
                    "info": {
                        "id": "msg_test",
                        "finish": "stop",
                        "tokens": {
                            "input": 10,
                            "output": 3,
                        },
                    },
                    "parts": [
                        {
                            "type": "text",
                            "text": "MUSE_SHIM_OK",
                        }
                    ],
                }
            )
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


def test_openai_shim_translates_to_opencode_session_api() -> None:
    module = _load_module()

    FakeOpenCodeHandler.requests = []
    opencode_server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        FakeOpenCodeHandler,
    )
    opencode_thread = threading.Thread(
        target=opencode_server.serve_forever,
        daemon=True,
    )
    opencode_thread.start()

    shim = module.create_server(
        host="127.0.0.1",
        port=0,
        opencode_url=(
            f"http://127.0.0.1:{opencode_server.server_address[1]}"
        ),
        provider_id="opencode",
        model_id="muse-spark-1.3-contributor-free",
        timeout_seconds=5,
    )
    shim_thread = threading.Thread(
        target=shim.serve_forever,
        daemon=True,
    )
    shim_thread.start()

    try:
        payload = {
            "model": "muse-spark-1.3-contributor-free",
            "messages": [
                {
                    "role": "system",
                    "content": "Use only supplied evidence.",
                },
                {
                    "role": "user",
                    "content": "Summarize the community.",
                },
            ],
            "temperature": 0,
        }
        request = urllib.request.Request(
            (
                "http://127.0.0.1:"
                f"{shim.server_address[1]}"
                "/v1/chat/completions"
            ),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
    finally:
        shim.shutdown()
        shim.server_close()
        opencode_server.shutdown()
        opencode_server.server_close()
        shim_thread.join(timeout=5)
        opencode_thread.join(timeout=5)

    assert result["model"] == "muse-spark-1.3-contributor-free"
    assert result["choices"][0]["message"]["content"] == "MUSE_SHIM_OK"
    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }

    message_request = FakeOpenCodeHandler.requests[1]
    assert message_request["path"] == "/session/ses_test/message"
    assert message_request["payload"]["model"] == {
        "providerID": "opencode",
        "modelID": "muse-spark-1.3-contributor-free",
    }
    assert message_request["payload"]["system"] == "Use only supplied evidence."
    assert (
        message_request["payload"]["parts"][0]["text"]
        == "USER: Summarize the community."
    )
