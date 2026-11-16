"""Parser-independent representation of a parsed document."""

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BBox:
    """A rectangle in page-image pixels, origin at the top-left corner."""

    x0: float
    y0: float
    x1: float
    y1: float

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 1) for k, v in vars(self).items()}


class ElementKind(enum.StrEnum):
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


@dataclass
class Element:
    """One block of a document, in reading order."""

    kind: ElementKind
    text: str
    page_number: int | None = None
    bbox: BBox | None = None
    # Depth of a heading; 1 is a top-level section.
    level: int = 1
    caption: str = ""
    # PNG crop for tables and figures.
    image: bytes | None = None


@dataclass
class PageRender:
    page_number: int
    width: int
    height: int
    image: bytes


@dataclass
class ParsedDocument:
    pages: list[PageRender] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)


class Parser(Protocol):
    def parse(self, path: Path) -> ParsedDocument:
        """Parse a file. Blocking and CPU-bound; call it from a worker thread."""
        ...
