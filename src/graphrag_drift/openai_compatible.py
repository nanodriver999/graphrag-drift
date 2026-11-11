from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from urllib import error, request


_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class OpenAICompatibleTextGenerator:
    """Minimal OpenAI-compatible chat-completions adapter with retry/fallback."""

    base_url: str
    model: str
    fallback_models: tuple[str, ...] = ()
    api_key: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5
    _successful_models: dict[str, int] = field(default_factory=dict, init=False, repr=False, compare=False)
    _model_lock: Any = field(default_factory=Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

    def reset_model_usage(self) -> None:
        """Clear per-model successful response counters for a new benchmark case."""
        with self._model_lock:
            self._successful_models.clear()

    def successful_model_usage(self) -> dict[str, int]:
        """Return successful chat replies by the actual requested model ID.

        Failed provider attempts and transport retries are excluded.
        """
        with self._model_lock:
            return dict(self._successful_models)

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2 ** attempt)
        if delay > 0:
            time.sleep(delay)

    def _request_for_model(self, prompt: str, model: str) -> request.Request:
        payload = {
            "model": model,
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
        return request.Request(
            self._endpoint(),
            data=body,
            headers=headers,
            method="POST",
        )

    def _request_model(self, prompt: str, model: str) -> dict:
        req = self._request_for_model(prompt, model)

        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                retryable = exc.code in _RETRYABLE_HTTP_STATUS
                if not retryable or attempt >= self.max_retries:
                    raise
                self._sleep_before_retry(attempt)
            except (error.URLError, TimeoutError):
                if attempt >= self.max_retries:
                    raise
                self._sleep_before_retry(attempt)

        raise RuntimeError("OpenAI-compatible request exhausted retries")

    def generate(self, prompt: str) -> str:
        models = [self.model, *self.fallback_models]
        last_error: Exception | None = None

        for model in models:
            try:
                data = self._request_model(prompt, model)
            except (error.HTTPError, error.URLError, TimeoutError) as exc:
                last_error = exc
                continue

            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    "OpenAI-compatible response is missing choices[0].message.content"
                ) from exc

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "OpenAI-compatible response content must be non-empty text"
                )
            with self._model_lock:
                self._successful_models[model] = (
                    self._successful_models.get(model, 0) + 1
                )
            return content

        if last_error is not None:
            raise last_error
        raise RuntimeError("No OpenAI-compatible model was available")
