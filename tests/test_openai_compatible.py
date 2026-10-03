from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from graphrag_drift.openai_compatible import OpenAICompatibleTextGenerator


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_generator_posts_chat_completion() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000",
        model="test-model",
        api_key="secret",
        timeout_seconds=12,
        temperature=0.1,
    )

    response = FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": "grounded answer",
                    }
                }
            ]
        }
    )

    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        return_value=response,
    ) as urlopen:
        text = generator.generate("summarize this")

    assert text == "grounded answer"

    req = urlopen.call_args.args[0]
    assert req.full_url == "http://localhost:8000/v1/chat/completions"
    assert req.get_header("Authorization") == "Bearer secret"
    assert req.get_header("Content-type") == "application/json"

    body = json.loads(req.data.decode("utf-8"))
    assert body == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "summarize this"}],
        "temperature": 0.1,
    }
    assert urlopen.call_args.kwargs["timeout"] == 12


def test_openai_compatible_generator_allows_no_api_key() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000/",
        model="local-model",
    )

    response = FakeResponse(
        {"choices": [{"message": {"content": "local answer"}}]}
    )

    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        return_value=response,
    ) as urlopen:
        generator.generate("hello")

    req = urlopen.call_args.args[0]
    assert req.full_url == "http://localhost:8000/v1/chat/completions"
    assert req.get_header("Authorization") is None


def test_openai_compatible_generator_rejects_invalid_response() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000",
        model="test-model",
    )

    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        return_value=FakeResponse({"choices": []}),
    ):
        with pytest.raises(ValueError, match="choices"):
            generator.generate("hello")
