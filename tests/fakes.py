"""Test doubles for the parser and model providers."""

from collections.abc import Sequence
from pathlib import Path

from intelliquery.config import EMBEDDING_DIMENSIONS
from intelliquery.ingestion.types import BBox, Element, ElementKind, PageRender, ParsedDocument

PNG = b"\x89PNG\r\n\x1a\n-not-a-real-image"


class FakeEmbedder:
    """Deterministic embedder: one-hot on the text length, so vectors are distinguishable."""

    model = "fake-embedder"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for value in texts:
            vector = [0.0] * EMBEDDING_DIMENSIONS
            vector[len(value) % EMBEDDING_DIMENSIONS] = 1.0
            vectors.append(vector)
        return vectors


class FakeDescriber:
    model = "fake-vision"

    def __init__(self, decorative_captions: Sequence[str] = ()) -> None:
        self._decorative = set(decorative_captions)

    async def describe(self, image_png: bytes, *, caption: str, section: str) -> str | None:
        return None if caption in self._decorative else f"A diagram in {section or 'the document'}."


class FakeParser:
    def __init__(self, document: ParsedDocument | None = None, *, fail_times: int = 0) -> None:
        self._document = document or sample_document()
        self._fail_times = fail_times
        self.calls = 0

    def parse(self, path: Path) -> ParsedDocument:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("parser exploded")
        return self._document


def sample_document() -> ParsedDocument:
    box = BBox(10, 20, 200, 60)
    return ParsedDocument(
        pages=[PageRender(1, 800, 1000, PNG), PageRender(2, 800, 1000, PNG)],
        elements=[
            Element(ElementKind.HEADING, "Installation", 1, box),
            Element(ElementKind.TEXT, "Mount the base station on a wall.", 1, box),
            Element(ElementKind.FIGURE, "", 1, box, caption="Figure 1: Wiring", image=PNG),
            Element(ElementKind.FIGURE, "", 1, box, caption="logo", image=PNG),
            Element(ElementKind.HEADING, "Alarms", 2, box),
            Element(ElementKind.TABLE, "| Limit | Value |\n|---|---|\n| Upper | 8 C |", 2, box),
        ],
    )
