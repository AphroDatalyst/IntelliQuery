import json
from typing import Any

import httpx
import pytest

from intelliquery.config import EMBEDDING_DIMENSIONS
from intelliquery.providers import ProviderError
from intelliquery.providers.openai_compat import (
    OpenAICompatClient,
    OpenAICompatEmbedder,
    OpenAICompatFigureDescriber,
)


def make_client(handler: Any, **kwargs: Any) -> OpenAICompatClient:
    return OpenAICompatClient(
        "http://models.test/v1", "secret", transport=httpx.MockTransport(handler), **kwargs
    )


def embedding_response(request: httpx.Request, dim: int = EMBEDDING_DIMENSIONS) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    # Returned in reverse to prove the client orders by `index`.
    data = [{"index": i, "embedding": [float(len(t))] * dim} for i, t in enumerate(inputs)]
    return httpx.Response(200, json={"data": data[::-1]})


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr("intelliquery.providers.openai_compat.asyncio.sleep", instant)


async def test_embedder_batches_and_preserves_input_order() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return embedding_response(request)

    embedder = OpenAICompatEmbedder(make_client(handler), "m", batch_size=2)
    vectors = await embedder.embed(["a", "bb", "ccc"])

    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]
    assert [len(json.loads(r.content)["input"]) for r in requests] == [2, 1]
    assert requests[0].url == "http://models.test/v1/embeddings"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert "dimensions" not in json.loads(requests[0].content)


async def test_embedder_can_request_the_index_dimension() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return embedding_response(request)

    await OpenAICompatEmbedder(make_client(handler), "m", request_dimensions=True).embed(["a"])
    assert seen["dimensions"] == EMBEDDING_DIMENSIONS


async def test_embedder_rejects_vectors_of_the_wrong_size() -> None:
    embedder = OpenAICompatEmbedder(make_client(lambda r: embedding_response(r, dim=768)), "m")
    with pytest.raises(ProviderError, match="768-dimensional"):
        await embedder.embed(["a"])


async def test_transient_failures_are_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("refused")
        if attempts == 2:
            return httpx.Response(503, text="overloaded")
        return embedding_response(request)

    vectors = await OpenAICompatEmbedder(make_client(handler, max_retries=3), "m").embed(["a"])
    assert attempts == 3
    assert len(vectors) == 1


async def test_retries_are_bounded() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, text="slow down")

    with pytest.raises(ProviderError, match="after 3 attempts"):
        await make_client(handler, max_retries=2).post("embeddings", {})
    assert attempts == 3


async def test_client_errors_are_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(ProviderError, match="HTTP 401"):
        await make_client(handler).post("embeddings", {})
    assert attempts == 1


def chat_handler(reply: str, seen: dict[str, Any] | None = None) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return handler


async def test_describer_sends_the_image_and_context() -> None:
    seen: dict[str, Any] = {}
    describer = OpenAICompatFigureDescriber(make_client(chat_handler(" A bar chart. ", seen)), "v")

    result = await describer.describe(b"\x89PNG", caption="Figure 2", section="Results")

    assert result == "A bar chart."
    assert seen["temperature"] == 0
    text_part, image_part = seen["messages"][1]["content"]
    assert "Figure 2" in text_part["text"]
    assert "Results" in text_part["text"]
    assert image_part["image_url"]["url"] == "data:image/png;base64,iVBORw=="


@pytest.mark.parametrize("reply", ["DECORATIVE", "decorative.", "", "   "])
async def test_describer_returns_none_for_decorative_or_empty(reply: str) -> None:
    describer = OpenAICompatFigureDescriber(make_client(chat_handler(reply)), "v")
    assert await describer.describe(b"x", caption="", section="") is None
