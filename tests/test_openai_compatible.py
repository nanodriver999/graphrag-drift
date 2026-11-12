from __future__ import annotations

import json
from io import BytesIO
from urllib.error import HTTPError, URLError
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
    assert generator.successful_model_usage() == {"test-model": 1}

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



def test_openai_compatible_generator_retries_transient_http_error() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000",
        model="test-model",
        max_retries=2,
        retry_backoff_seconds=0,
    )
    transient = HTTPError(
        "http://localhost:8000/v1/chat/completions",
        502,
        "Bad Gateway",
        hdrs=None,
        fp=None,
    )
    response = FakeResponse(
        {"choices": [{"message": {"content": "recovered answer"}}]}
    )

    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        side_effect=[transient, response],
    ) as urlopen:
        assert generator.generate("hello") == "recovered answer"

    assert urlopen.call_count == 2


def test_openai_compatible_generator_does_not_retry_permanent_http_error() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000",
        model="test-model",
        max_retries=3,
        retry_backoff_seconds=0,
    )
    permanent = HTTPError(
        "http://localhost:8000/v1/chat/completions",
        400,
        "Bad Request",
        hdrs=None,
        fp=None,
    )

    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        side_effect=permanent,
    ) as urlopen:
        with pytest.raises(HTTPError):
            generator.generate("hello")

    assert urlopen.call_count == 1


def test_openai_compatible_generator_retries_url_error() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000",
        model="test-model",
        max_retries=1,
        retry_backoff_seconds=0,
    )
    response = FakeResponse(
        {"choices": [{"message": {"content": "network recovered"}}]}
    )

    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        side_effect=[URLError("temporary"), response],
    ) as urlopen:
        assert generator.generate("hello") == "network recovered"

    assert urlopen.call_count == 2


@pytest.mark.parametrize(
    ("max_retries", "retry_backoff_seconds"),
    [(-1, 0), (1, -0.1)],
)
def test_openai_compatible_generator_rejects_invalid_retry_settings(
    max_retries: int,
    retry_backoff_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleTextGenerator(
            base_url="http://localhost:8000",
            model="test-model",
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )


def test_openai_compatible_generator_retries_timeout_error() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000",
        model="test-model",
        max_retries=1,
        retry_backoff_seconds=0,
    )
    response = FakeResponse(
        {"choices": [{"message": {"content": "timeout recovered"}}]}
    )

    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        side_effect=[TimeoutError("slow provider"), response],
    ) as urlopen:
        assert generator.generate("hello") == "timeout recovered"

    assert urlopen.call_count == 2


def test_openai_compatible_generator_falls_back_after_timeout_exhaustion() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000",
        model="primary",
        fallback_models=("opencode/fallback",),
        max_retries=0,
        retry_backoff_seconds=0,
    )
    response = FakeResponse(
        {"choices": [{"message": {"content": "fallback answer"}}]}
    )

    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        side_effect=[TimeoutError("primary timeout"), response],
    ) as urlopen:
        assert generator.generate("hello") == "fallback answer"

    primary_body = json.loads(urlopen.call_args_list[0].args[0].data.decode("utf-8"))
    fallback_body = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
    assert primary_body["model"] == "primary"
    assert fallback_body["model"] == "opencode/fallback"

    assert generator.successful_model_usage() == {"opencode/fallback": 1}
    generator.reset_model_usage()
    assert generator.successful_model_usage() == {}



def test_openai_generator_reports_sanitized_gateway_root_cause() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000",
        model="beam-free",
        max_retries=0,
    )
    upstream = HTTPError(
        "http://localhost:8000/v1/chat/completions",
        502,
        "Bad Gateway",
        hdrs=None,
        fp=BytesIO(json.dumps({
            "error": {
                "type": "upstream_error",
                "message": "OpenCode HTTP 429: provider rate limit. api_key=secret-token",
            }
        }).encode("utf-8")),
    )
    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        side_effect=upstream,
    ):
        with pytest.raises(HTTPError, match="OpenCode HTTP 429") as exc:
            generator.generate("detailed legal primer")
    assert "rate limit" in str(exc.value)
    assert "secret-token" not in str(exc.value)
    assert "api_key=[REDACTED]" in str(exc.value)


def test_openai_generator_does_not_require_error_body_for_gateway_failure() -> None:
    generator = OpenAICompatibleTextGenerator(
        base_url="http://localhost:8000", model="beam-free", max_retries=0,
    )
    upstream = HTTPError(
        "http://localhost:8000/v1/chat/completions",
        502, "Bad Gateway", hdrs=None, fp=None,
    )
    with patch(
        "graphrag_drift.openai_compatible.request.urlopen",
        side_effect=upstream,
    ):
        with pytest.raises(HTTPError, match="Bad Gateway"):
            generator.generate("test")
