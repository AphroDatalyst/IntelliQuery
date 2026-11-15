"""Exercise the migrated schema against a real PostgreSQL + pgvector."""

import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from intelliquery.config import EMBEDDING_DIMENSIONS
from intelliquery.db import models
from intelliquery.db.session import check_database

pytestmark = pytest.mark.integration


def _unit_vector(axis: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[axis] = 1.0
    return vector


async def _seed_document(session: AsyncSession) -> models.Document:
    user = models.User(email=f"{uuid.uuid4().hex}@example.com", hashed_password="x")
    collection = models.Collection(name="manuals", owner=user)
    document = models.Document(
        collection=collection,
        filename="manual.pdf",
        content_type="application/pdf",
        sha256=uuid.uuid4().hex * 2,
        size_bytes=1,
        storage_key="k",
    )
    session.add(document)
    await session.flush()
    return document


async def test_database_reports_pgvector(engine: AsyncEngine) -> None:
    versions = await check_database(engine)
    assert versions["pgvector"]


async def test_vector_and_fulltext_search(engine: AsyncEngine) -> None:
    async with AsyncSession(engine) as session:
        document = await _seed_document(session)
        session.add_all(
            [
                models.Chunk(
                    document_id=document.id,
                    collection_id=document.collection_id,
                    chunk_index=0,
                    modality=models.ChunkModality.TEXT,
                    content="The logger stores temperature readings every minute.",
                    embedding=_unit_vector(0),
                ),
                models.Chunk(
                    document_id=document.id,
                    collection_id=document.collection_id,
                    chunk_index=1,
                    modality=models.ChunkModality.FIGURE,
                    content="Wiring diagram of the base station.",
                    context="Installation chapter, humidity probe connections.",
                    embedding=_unit_vector(1),
                ),
            ]
        )
        await session.flush()

        nearest = await session.scalar(
            select(models.Chunk.chunk_index)
            .where(models.Chunk.collection_id == document.collection_id)
            .order_by(models.Chunk.embedding.cosine_distance(_unit_vector(1)))
            .limit(1)
        )
        assert nearest == 1

        # Stemming matches "readings" -> "reading"; the generated context is searchable too.
        by_content = await session.scalar(
            select(models.Chunk.chunk_index).where(
                models.Chunk.collection_id == document.collection_id,
                models.Chunk.search_vector.op("@@")(func.plainto_tsquery("english", "reading")),
            )
        )
        by_context = await session.scalar(
            select(models.Chunk.chunk_index).where(
                models.Chunk.collection_id == document.collection_id,
                models.Chunk.search_vector.op("@@")(func.plainto_tsquery("english", "humidity")),
            )
        )
        assert (by_content, by_context) == (0, 1)
        await session.rollback()


async def test_duplicate_upload_is_rejected(engine: AsyncEngine) -> None:
    async with AsyncSession(engine) as session:
        document = await _seed_document(session)
        session.add(
            models.Document(
                collection_id=document.collection_id,
                filename="copy.pdf",
                content_type="application/pdf",
                sha256=document.sha256,
                size_bytes=1,
                storage_key="k2",
            )
        )
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()


async def test_deleting_a_document_removes_its_chunks(engine: AsyncEngine) -> None:
    async with AsyncSession(engine) as session:
        document = await _seed_document(session)
        session.add(
            models.Chunk(
                document_id=document.id,
                collection_id=document.collection_id,
                chunk_index=0,
                modality=models.ChunkModality.TEXT,
                content="x",
            )
        )
        await session.flush()
        await session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document.id})
        remaining = await session.scalar(
            select(func.count())
            .select_from(models.Chunk)
            .where(models.Chunk.document_id == document.id)
        )
        assert remaining == 0
        await session.rollback()
