"""Providers for any endpoint that implements the OpenAI-compatible HTTP API."""

import asyncio
import base64
import logging
import random
from collections.abc import Sequence
from typing import Any

import httpx

from intelliquery.config import EMBEDDING_DIMENSIONS
from intelliquery.providers import ChatMessage, ChatResult, ProviderError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

DECORATIVE = "DECORATIVE"

FIGURE_PROMPT = f"""You are indexing a figure from a document so it can be found by search \
and cited as evidence.

Describe only what is visible in the image:
- what kind of figure it is (chart, diagram, screenshot, photo, schematic...)
- every piece of readable text, label, axis title, legend entry and number, transcribed exactly
- the relationships it shows (trends, connections, sequence, layout)

Do not interpret beyond the image, do not guess unreadable values, and do not add outside \
knowledge. If part of the image is illegible, say so.

If the image is purely decorative (a logo, icon, divider or background), reply with exactly \
{DECORATIVE} and nothing else."""


class OpenAICompatClient:
    """Thin JSON-over-HTTP client with bounded retries on transient failures."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self._max_retries = max_retries

    async def aclose(self) -> None:
        await self._http.aclose()

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error = "no attempt made"
        for attempt in range(self._max_retries + 1):
            if attempt:
                # Exponential backoff with jitter, capped so a job never stalls for minutes.
                await asyncio.sleep(min(2.0**attempt, 20.0) * random.uniform(0.5, 1.0))  # noqa: S311
            try:
                response = await self._http.post(path, json=payload)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("model request failed", extra={"path": path, "error": last_error})
                continue
            if response.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning("model request retryable", extra={"path": path, "error": last_error})
                continue
            if response.is_error:
                raise ProviderError(f"{path} -> HTTP {response.status_code}: {response.text[:500]}")
            body = response.json()
            if not isinstance(body, dict):
                raise ProviderError(f"{path} returned a non-object response")
            return body
        raise ProviderError(f"{path} failed after {self._max_retries + 1} attempts: {last_error}")


class OpenAICompatEmbedder:
    def __init__(
        self,
        client: OpenAICompatClient,
        model: str,
        *,
        batch_size: int = 32,
        request_dimensions: bool = False,
    ) -> None:
        self._client = client
        self.model = model
        self._batch_size = batch_size
        self._request_dimensions = request_dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            payload: dict[str, Any] = {"model": self.model, "input": batch}
            if self._request_dimensions:
                payload["dimensions"] = EMBEDDING_DIMENSIONS
            body = await self._client.post("embeddings", payload)
            try:
                # The API does not guarantee order; `index` does.
                rows = sorted(body["data"], key=lambda row: row["index"])
                batch_vectors = [[float(x) for x in row["embedding"]] for row in rows]
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError(f"malformed embeddings response: {exc!r}") from exc
            if len(batch_vectors) != len(batch):
                raise ProviderError(
                    f"asked for {len(batch)} embeddings, received {len(batch_vectors)}"
                )
            for vector in batch_vectors:
                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise ProviderError(
                        f"model {self.model!r} returned {len(vector)}-dimensional vectors; "
                        f"the index is built for {EMBEDDING_DIMENSIONS}"
                    )
            vectors.extend(batch_vectors)
        return vectors


class OpenAICompatFigureDescriber:
    def __init__(self, client: OpenAICompatClient, model: str, *, max_tokens: int = 700) -> None:
        self._client = client
        self.model = model
        self._max_tokens = max_tokens

    async def describe(self, image_png: bytes, *, caption: str, section: str) -> str | None:
        hints = []
        if section:
            hints.append(f"Document section: {section}")
        if caption:
            hints.append(f"Printed caption: {caption}")
        data_uri = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        body = await self._client.post(
            "chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": FIGURE_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "\n".join(hints) or "Describe this figure."},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    },
                ],
            },
        )
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed chat response: {exc!r}") from exc
        text = (text or "").strip()
        if not text or text.upper().startswith(DECORATIVE):
            return None
        return str(text)


def _png_part(image: bytes) -> dict[str, Any]:
    data_uri = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    return {"type": "image_url", "image_url": {"url": data_uri}}


class OpenAICompatChat:
    def __init__(self, client: OpenAICompatClient, model: str) -> None:
        self._client = client
        self.model = model

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, json_output: bool = False
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            # Answers must be reproducible and stay close to the sources.
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": message.role,
                    "content": [
                        {"type": "text", "text": message.content},
                        *(_png_part(image) for image in message.images),
                    ]
                    if message.images
                    else message.content,
                }
                for message in messages
            ],
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        body = await self._client.post("chat/completions", payload)
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed chat response: {exc!r}") from exc
        usage = body.get("usage") or {}
        return ChatResult(
            text=str(text).strip(),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


class HttpReranker:
    """Cross-encoder reranker behind a Cohere/Jina-style `POST /rerank` endpoint."""

    def __init__(self, client: OpenAICompatClient, model: str) -> None:
        self._client = client
        self.model = model

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        body = await self._client.post(
            "rerank",
            {
                "model": self.model,
                "query": query,
                "documents": list(documents),
                "top_n": len(documents),
            },
        )
        scores = [0.0] * len(documents)
        try:
            for row in body["results"]:
                scores[int(row["index"])] = float(row["relevance_score"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"malformed rerank response: {exc!r}") from exc
        return scores
