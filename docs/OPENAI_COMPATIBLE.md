# OpenAI-compatible text generator

`OpenAICompatibleTextGenerator` implements the provider-neutral `TextGenerator`
interface used by `GlobalLLMReasoner`.

It calls:

```text
POST <base_url>/v1/chat/completions
```

without depending on a provider SDK.

This makes the same GraphRAG Global Search reasoner usable with:

- an OpenAI-compatible hosted endpoint;
- LiteLLM;
- vLLM;
- another service that implements the same chat-completions request/response shape.

## Example

```python
from graphrag_drift import (
    GlobalLLMReasoner,
    OpenAICompatibleTextGenerator,
)

generator = OpenAICompatibleTextGenerator(
    base_url="http://localhost:8000",
    model="my-model",
    api_key=None,
)

reasoner = GlobalLLMReasoner(generator=generator)
```

For hosted providers, pass the API key from an environment variable or secret
manager. Do not hard-code credentials into source files.

## Scope

This adapter only performs text generation.

Selecting a provider, reading environment configuration, retries/backoff,
structured output, and live-provider integration tests are separate concerns.
