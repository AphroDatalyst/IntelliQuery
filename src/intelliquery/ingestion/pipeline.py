"""Turn a stored file into searchable, citable chunks."""

import asyncio
import logging
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelliquery.db import models
from intelliquery.db.models import ChunkModality, DocumentStatus
from intelliquery.ingestion.chunking import ChunkDraft, build_chunks
from intelliquery.ingestion.service import asset_key, page_key
from intelliquery.ingestion.types import Parser
from intelliquery.providers import Embedder, FigureDescriber
from intelliquery.storage import ObjectStorage

logger = logging.getLogger(__name__)

StageCallback = Callable[[str], Awaitable[None]]


@dataclass
class IngestionResult:
    pages: int
    chunks: int
    figures_described: int
    figures_dropped: int


async def _noop(_stage: str) -> None:
    return None


class IngestionPipeline:
    def __init__(
        self,
        *,
        parser: Parser,
        embedder: Embedder,
        storage: ObjectStorage,
        describer: FigureDescriber | None = None,
        max_chunk_tokens: int = 350,
        vision_concurrency: int = 4,
    ) -> None:
        self._parser = parser
        self._embedder = embedder
        self._describer = describer
        self._storage = storage
        self._max_chunk_tokens = max_chunk_tokens
        self._vision_limit = asyncio.Semaphore(vision_concurrency)

    async def _describe(self, draft: ChunkDraft) -> bool:
        """Fill in a figure's description. Returns False when the figure should be dropped."""
        if self._describer is None or draft.image is None:
            # Without a description, a figure is only findable through its printed caption.
            return bool(draft.caption)
        async with self._vision_limit:
            description = await self._describer.describe(
                draft.image, caption=draft.caption, section=draft.section_path
            )
        if description is None:
            return False
        draft.content = "\n\n".join(part for part in (draft.caption, description) if part)
        return True

    async def run(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        document_id: uuid.UUID,
        on_stage: StageCallback = _noop,
    ) -> IngestionResult:
        async with session_factory() as session:
            document = await session.get(models.Document, document_id)
            if document is None:
                raise LookupError(f"document {document_id} no longer exists")
            storage_key, collection_id = document.storage_key, document.collection_id

        await on_stage("parsing")
        source = await self._storage.get(storage_key)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.pdf"
            path.write_bytes(source)
            parsed = await asyncio.to_thread(self._parser.parse, path)

        drafts = build_chunks(parsed.elements, max_tokens=self._max_chunk_tokens)

        await on_stage("describing figures")
        figures = [d for d in drafts if d.modality is ChunkModality.FIGURE]
        kept = await asyncio.gather(*(self._describe(d) for d in figures))
        dropped = {id(d) for d, keep in zip(figures, kept, strict=True) if not keep}
        drafts = [d for d in drafts if id(d) not in dropped]
        if not drafts:
            raise ValueError("no extractable content; the file may be scanned (enable OCR)")

        await on_stage("embedding")
        vectors = await self._embedder.embed([d.retrieval_text for d in drafts])

        await on_stage("storing")
        for page in parsed.pages:
            await self._storage.put(page_key(document_id, page.page_number), page.image)
        for index, draft in enumerate(drafts):
            if draft.image is not None:
                await self._storage.put(asset_key(document_id, index), draft.image)

        # One transaction: a re-run replaces the previous result, and readers never
        # see a half-indexed document.
        async with session_factory() as session, session.begin():
            await session.execute(
                delete(models.Chunk).where(models.Chunk.document_id == document_id)
            )
            await session.execute(
                delete(models.DocumentPage).where(models.DocumentPage.document_id == document_id)
            )
            session.add_all(
                models.DocumentPage(
                    document_id=document_id,
                    page_number=page.page_number,
                    image_key=page_key(document_id, page.page_number),
                    width=page.width,
                    height=page.height,
                )
                for page in parsed.pages
            )
            session.add_all(
                models.Chunk(
                    document_id=document_id,
                    collection_id=collection_id,
                    chunk_index=index,
                    modality=draft.modality,
                    page_number=draft.page_number,
                    bbox=draft.bbox.as_dict() if draft.bbox else None,
                    section_path=draft.section_path or None,
                    content=draft.content,
                    # Headings are indexed for keyword search alongside the content.
                    context=draft.section_path or None,
                    asset_key=asset_key(document_id, index) if draft.image else None,
                    token_count=draft.token_count,
                    embedding=vector,
                    embedding_model=self._embedder.model,
                    meta={"regions": draft.regions},
                )
                for index, (draft, vector) in enumerate(zip(drafts, vectors, strict=True))
            )
            document = await session.get(models.Document, document_id)
            if document is None:
                raise LookupError(f"document {document_id} was deleted during ingestion")
            document.status = DocumentStatus.READY
            document.error = None
            document.page_count = len(parsed.pages) or None

        result = IngestionResult(
            pages=len(parsed.pages),
            chunks=len(drafts),
            figures_described=sum(kept) if self._describer else 0,
            figures_dropped=len(dropped),
        )
        logger.info("ingested document", extra={"document_id": str(document_id), **vars(result)})
        return result
