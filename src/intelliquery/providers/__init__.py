"""Model provider interfaces.

The pipeline depends on these protocols only, so hosted and local models are
interchangeable by configuration.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol


class ProviderError(RuntimeError):
    """A model endpoint failed or returned something unusable."""


class Embedder(Protocol):
    model: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FigureDescriber(Protocol):
    model: str

    async def describe(self, image_png: bytes, *, caption: str, section: str) -> str | None:
        """Describe a figure, or return None when it carries no information."""
        ...


class Reranker(Protocol):
    model: str

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document, in the order given."""
        ...


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str
    # PNG images attached to the message, for vision-capable models.
    images: list[bytes] = field(default_factory=list)


@dataclass
class ChatResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatModel(Protocol):
    model: str

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, json_output: bool = False
    ) -> ChatResult: ...
