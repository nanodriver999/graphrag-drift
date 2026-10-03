from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request


@dataclass(frozen=True)
class OpenAICompatibleTextGenerator:
    """Minimal OpenAI-compatible chat-completions adapter.

    This intentionally uses only the Python standard library so the GraphRAG
    package does not need a provider-specific SDK. It can target OpenAI,
    LiteLLM, vLLM, or another compatible endpoint.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            self._endpoint(),
            data=body,
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "OpenAI-compatible response is missing choices[0].message.content"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenAI-compatible response content must be non-empty text")

        return content
