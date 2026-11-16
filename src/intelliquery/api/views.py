"""Building API responses for conversations: messages with their citations."""

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from intelliquery.api.schemas import Box, CitationOut, MessageOut, Region
from intelliquery.db import models
from intelliquery.db.models import MessageRole


def _box(raw: Any) -> Box | None:
    try:
        return Box(**raw) if raw else None
    except (TypeError, ValueError):
        return None


def _chunk_regions(chunk: models.Chunk) -> list[tuple[int, Box | None]]:
    """Pages a chunk covers, with its box on each; a chunk may continue onto a second page."""
    regions: list[tuple[int, Box | None]] = []
    for region in chunk.meta.get("regions") or []:
        if isinstance(region, dict) and isinstance(region.get("page"), int):
            regions.append((region["page"], _box(region.get("bbox"))))
    if not regions and chunk.page_number is not None:
        regions.append((chunk.page_number, _box(chunk.bbox)))
    return regions


async def message_views(
    session: AsyncSession, user_id: uuid.UUID, messages: Sequence[models.Message]
) -> list[MessageOut]:
    """Render messages, in the order given. Citations must already be loaded."""
    chunk_ids = {c.chunk_id for m in messages for c in m.citations}
    chunks: dict[uuid.UUID, tuple[models.Chunk, str]] = {}
    if chunk_ids:
        rows = await session.execute(
            select(models.Chunk, models.Document.filename)
            .join(models.Document, models.Document.id == models.Chunk.document_id)
            .where(models.Chunk.id.in_(chunk_ids))
        )
        chunks = {chunk.id: (chunk, filename) for chunk, filename in rows}

    regions = {chunk_id: _chunk_regions(chunk) for chunk_id, (chunk, _) in chunks.items()}
    wanted = {(chunks[cid][0].document_id, page) for cid, rs in regions.items() for page, _ in rs}
    sizes: dict[tuple[uuid.UUID, int], tuple[int, int]] = {}
    if wanted:
        pages = await session.execute(
            select(
                models.DocumentPage.document_id,
                models.DocumentPage.page_number,
                models.DocumentPage.width,
                models.DocumentPage.height,
            ).where(
                tuple_(models.DocumentPage.document_id, models.DocumentPage.page_number).in_(wanted)
            )
        )
        sizes = {(doc, page): (width, height) for doc, page, width, height in pages}

    ratings: dict[uuid.UUID, int] = {}
    message_ids = [m.id for m in messages]
    if message_ids:
        votes = await session.execute(
            select(models.Feedback.message_id, models.Feedback.rating).where(
                models.Feedback.user_id == user_id, models.Feedback.message_id.in_(message_ids)
            )
        )
        ratings = {row.message_id: row.rating for row in votes}

    views = []
    for message in messages:
        citations = []
        for citation in sorted(message.citations, key=lambda c: c.marker):
            chunk, filename = chunks[citation.chunk_id]
            citations.append(
                CitationOut(
                    marker=citation.marker,
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    filename=filename,
                    modality=chunk.modality,
                    page_number=chunk.page_number,
                    section_path=chunk.section_path,
                    content=chunk.content,
                    has_image=chunk.asset_key is not None,
                    regions=[
                        Region(
                            page=page,
                            box=box,
                            page_width=sizes.get((chunk.document_id, page), (None, None))[0],
                            page_height=sizes.get((chunk.document_id, page), (None, None))[1],
                        )
                        for page, box in regions[chunk.id]
                    ],
                    supported=(
                        None if citation.support_score is None else citation.support_score >= 0.5
                    ),
                )
            )
        rating = ratings.get(message.id)
        views.append(
            MessageOut(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
                abstained=message.abstained,
                verified=bool(citations) and all(c.supported for c in citations),
                model=message.model,
                latency_ms=message.latency_ms,
                citations=citations,
                rating=1 if rating == 1 else -1 if rating == -1 else None,
            )
        )
    return views


async def conversation_messages(
    session: AsyncSession, conversation_id: uuid.UUID, *, last: int | None = None
) -> list[models.Message]:
    """Messages of a conversation, oldest first; `last` keeps only the most recent ones."""
    # Both messages of a turn share a transaction timestamp; the question sorts first.
    order = (models.Message.created_at, models.Message.role == MessageRole.ASSISTANT)
    query = (
        select(models.Message)
        .where(models.Message.conversation_id == conversation_id)
        .options(selectinload(models.Message.citations))
    )
    if last is None:
        return list(await session.scalars(query.order_by(*order)))
    newest = await session.scalars(query.order_by(*(o.desc() for o in order)).limit(last))
    return list(reversed(newest.all()))
