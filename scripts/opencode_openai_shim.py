#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") in {"text", "input_text"}
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    return ""


def _messages_to_opencode(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system_parts: list[str] = []
    prompt_parts: list[str] = []

    for message in messages:
        role = str(message.get("role", "user"))
        text = _text_from_content(message.get("content"))
        if not text:
            continue

        if role == "system":
            system_parts.append(text)
        else:
            prompt_parts.append(f"{role.upper()}: {text}")

    return "\n".join(system_parts), "\n\n".join(prompt_parts)


class OpenCodeClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> Any:
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")

        request = urllib.request.Request(
            self.base_url + path,
            data=payload,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(
            request,
            timeout=self.timeout_seconds,
        ) as response:
            raw = response.read()
            return json.loads(raw) if raw else None


class OpenAIShimServer(ThreadingHTTPServer):
    client: OpenCodeClient
    provider_id: str
    model_id: str


class Handler(BaseHTTPRequestHandler):
    server: OpenAIShimServer

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _parse_model(self, requested: str | None) -> tuple[str, str]:
        if not requested:
            return self.server.provider_id, self.server.model_id

        if "/" in requested:
            provider_id, model_id = requested.split("/", 1)
            return provider_id, model_id

        return self.server.provider_id, requested

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            try:
                health = self.server.client.request("/global/health")
            except Exception as exc:
                self._json_response(
                    503,
                    {"ok": False, "error": str(exc)},
                )
                return

            self._json_response(
                200,
                {"ok": True, "opencode": health},
            )
            return

        if self.path == "/v1/models":
            self._json_response(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_id,
                            "object": "model",
                            "owned_by": self.server.provider_id,
                        }
                    ],
                },
            )
            return

        self._json_response(
            404,
            {
                "error": {
                    "message": "not found",
                    "type": "not_found_error",
                }
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json_response(
                404,
                {
                    "error": {
                        "message": "not found",
                        "type": "not_found_error",
                    }
                },
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")

            if payload.get("stream"):
                self._json_response(
                    400,
                    {
                        "error": {
                            "message": "stream=true is not supported by this shim",
                            "type": "invalid_request_error",
                        }
                    },
                )
                return

            messages = payload.get("messages")
            if not isinstance(messages, list):
                self._json_response(
                    400,
                    {
                        "error": {
                            "message": "messages must be an array",
                            "type": "invalid_request_error",
                        }
                    },
                )
                return

            provider_id, model_id = self._parse_model(payload.get("model"))
            system, prompt = _messages_to_opencode(messages)

            session = self.server.client.request(
                "/session",
                method="POST",
                body={"title": "OpenAI compatibility request"},
            )

            body: dict[str, Any] = {
                "model": {
                    "providerID": provider_id,
                    "modelID": model_id,
                },
                "parts": [
                    {
                        "type": "text",
                        "text": prompt or " ",
                    }
                ],
                "tools": {},
            }
            if system:
                body["system"] = system

            result = self.server.client.request(
                f"/session/{session['id']}/message",
                method="POST",
                body=body,
            )

            content = "\n".join(
                part.get("text", "")
                for part in result.get("parts", [])
                if part.get("type") == "text" and part.get("text")
            )
            info = result.get("info", {})
            tokens = info.get("tokens", {})
            prompt_tokens = int(tokens.get("input") or 0)
            completion_tokens = int(tokens.get("output") or 0)

            self._json_response(
                200,
                {
                    "id": info.get("id")
                    or f"chatcmpl-opencode-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                            "finish_reason": info.get("finish") or "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            self._json_response(
                502,
                {
                    "error": {
                        "message": f"OpenCode HTTP {exc.code}: {detail}",
                        "type": "upstream_error",
                    }
                },
            )
        except Exception as exc:
            self._json_response(
                502,
                {
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                    }
                },
            )

    def log_message(self, *_args: object) -> None:
        return


def create_server(
    *,
    host: str,
    port: int,
    opencode_url: str,
    provider_id: str,
    model_id: str,
    timeout_seconds: float,
) -> OpenAIShimServer:
    server = OpenAIShimServer((host, port), Handler)
    server.client = OpenCodeClient(opencode_url, timeout_seconds)
    server.provider_id = provider_id
    server.model_id = model_id
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=os.environ.get("SHIM_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SHIM_PORT", "8000")),
    )
    parser.add_argument(
        "--opencode-url",
        default=os.environ.get(
            "OPENCODE_URL",
            "http://127.0.0.1:4096",
        ),
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get(
            "OPENCODE_PROVIDER_ID",
            "opencode",
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "OPENCODE_MODEL_ID",
            "muse-spark-1.3-contributor-free",
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(
            os.environ.get(
                "OPENCODE_REQUEST_TIMEOUT",
                "180",
            )
        ),
    )
    args = parser.parse_args()

    server = create_server(
        host=args.host,
        port=args.port,
        opencode_url=args.opencode_url,
        provider_id=args.provider,
        model_id=args.model,
        timeout_seconds=args.timeout,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
