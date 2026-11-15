"""Structure-aware chunking.

Chunks follow the document's own structure instead of a fixed character window:
text is grouped under its heading and never merged across a section boundary, and
each table or figure is a chunk of its own. Every chunk records where it sits on
the page so an answer can point at its evidence.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from intelliquery.db.models import ChunkModality
from intelliquery.ingestion.types import BBox, Element, ElementKind

_TOKEN = re.compile(r"\w+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SECTION_SEPARATOR = " > "


def estimate_tokens(text: str) -> int:
    """Cheap, model-independent token estimate (words plus punctuation)."""
    return len(_TOKEN.findall(text))


@dataclass
class ChunkDraft:
    modality: ChunkModality
    content: str
    section_path: str = ""
    page_number: int | None = None
    bbox: BBox | None = None
    # Every page region the chunk covers, for chunks that span blocks or pages.
    regions: list[dict[str, Any]] = field(default_factory=list)
    image: bytes | None = None
    caption: str = ""

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.content)

    @property
    def retrieval_text(self) -> str:
        """Text to embed: the content, situated by the headings above it."""
        return f"{self.section_path}\n\n{self.content}" if self.section_path else self.content


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    """Split an oversized block at sentence boundaries, falling back to words."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in _SENTENCE_END.split(text):
        tokens = estimate_tokens(sentence)
        if tokens > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current, current_tokens = [], 0
            words = sentence.split()
            step = max(1, max_tokens // 2)
            pieces.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
            continue
        if current and current_tokens + tokens > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _regions(elements: list[Element]) -> list[dict[str, Any]]:
    """Merge the elements' boxes into one region per page."""
    by_page: dict[int, BBox] = {}
    for element in elements:
        if element.page_number is None or element.bbox is None:
            continue
        existing = by_page.get(element.page_number)
        by_page[element.page_number] = existing.union(element.bbox) if existing else element.bbox
    return [{"page": page, "bbox": box.as_dict()} for page, box in sorted(by_page.items())]


def _text_chunk(elements: list[Element], section_path: str) -> ChunkDraft:
    regions = _regions(elements)
    first_page = next((e.page_number for e in elements if e.page_number is not None), None)
    first_box = next((r["bbox"] for r in regions if r["page"] == first_page), None)
    return ChunkDraft(
        modality=ChunkModality.TEXT,
        content="\n".join(e.text for e in elements),
        section_path=section_path,
        page_number=first_page,
        bbox=BBox(**first_box) if first_box else None,
        regions=regions,
    )


def build_chunks(elements: list[Element], *, max_tokens: int = 350) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    headings: list[tuple[int, str]] = []
    buffer: list[Element] = []
    buffer_tokens = 0

    def section_path() -> str:
        return SECTION_SEPARATOR.join(title for _, title in headings)

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append(_text_chunk(buffer, section_path()))
        buffer, buffer_tokens = [], 0

    for element in elements:
        text = element.text.strip()

        if element.kind is ElementKind.HEADING:
            if not text:
                continue
            flush()
            # A heading closes every open section at its own depth or deeper.
            while headings and headings[-1][0] >= element.level:
                headings.pop()
            headings.append((element.level, text))
            continue

        if element.kind in (ElementKind.TABLE, ElementKind.FIGURE):
            caption = element.caption.strip()
            content = "\n\n".join(part for part in (caption, text) if part)
            is_table = element.kind is ElementKind.TABLE
            if is_table and not content:
                continue
            flush()
            chunks.append(
                ChunkDraft(
                    modality=ChunkModality.TABLE if is_table else ChunkModality.FIGURE,
                    content=content,
                    section_path=section_path(),
                    page_number=element.page_number,
                    bbox=element.bbox,
                    regions=_regions([element]),
                    image=element.image,
                    caption=caption,
                )
            )
            continue

        if not text:
            continue
        tokens = estimate_tokens(text)
        if tokens > max_tokens:
            flush()
            for piece in _split_long_text(text, max_tokens):
                part = Element(ElementKind.TEXT, piece, element.page_number, element.bbox)
                chunks.append(_text_chunk([part], section_path()))
            continue
        if buffer_tokens + tokens > max_tokens:
            flush()
        buffer.append(Element(ElementKind.TEXT, text, element.page_number, element.bbox))
        buffer_tokens += tokens

    flush()
    return chunks
