"""Model provider interfaces.

The pipeline depends on these protocols only, so hosted and local models are
interchangeable by configuration.
"""

from collections.abc import Sequence
from typing import Protocol


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
