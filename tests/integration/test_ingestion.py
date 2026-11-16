"""Ingestion against a real database, with the parser and models faked."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelliquery.cli import get_or_create_collection
from intelliquery.config import Settings
from intelliquery.db import models
from intelliquery.db.models import ChunkModality, DocumentStatus, JobStatus
from intelliquery.ingestion import jobs
from intelliquery.ingestion.pipeline import IngestionPipeline
from intelliquery.ingestion.service import page_key, register_document
from intelliquery.storage import LocalStorage
from intelliquery.worker import run_next_job
from tests.fakes import PNG, FakeDescriber, FakeEmbedder, FakeParser

pytestmark = pytest.mark.integration

PDF = b"%PDF-1.7 sample"
STALE = timedelta(minutes=30)
Factory = async_sessionmaker[AsyncSession]


async def upload(factory: Factory, storage: LocalStorage, data: bytes = PDF) -> uuid.UUID:
    async with factory() as session, session.begin():
        collection = await get_or_create_collection(session, "manuals")
        document, _created = await register_document(
            session,
            storage,
            collection_id=collection.id,
            filename="manual.pdf",
            data=data,
            max_bytes=10_000,
        )
        return document.id


def make_pipeline(storage: LocalStorage, parser: FakeParser | None = None) -> IngestionPipeline:
    return IngestionPipeline(
        parser=parser or FakeParser(),
        embedder=FakeEmbedder(),
        describer=FakeDescriber(decorative_captions=["logo"]),
        storage=storage,
    )


async def test_upload_is_stored_queued_and_deduplicated(
    session_factory: Factory, storage: LocalStorage
) -> None:
    first = await upload(session_factory, storage)
    second = await upload(session_factory, storage)
    other = await upload(session_factory, storage, PDF + b" v2")

    assert first == second != other
    async with session_factory() as session:
        document = await session.get_one(models.Document, first)
        assert await storage.get(document.storage_key) == PDF
        assert document.status is DocumentStatus.PENDING
        queued = await session.scalar(select(func.count()).select_from(models.IngestionJob))
        assert queued == 2


async def test_worker_ingests_a_document_end_to_end(
    session_factory: Factory, storage: LocalStorage, settings: Settings
) -> None:
    document_id = await upload(session_factory, storage)

    assert await run_next_job(session_factory, make_pipeline(storage), settings) is True
    assert await run_next_job(session_factory, make_pipeline(storage), settings) is False

    async with session_factory() as session:
        document = await session.get_one(models.Document, document_id)
        assert (document.status, document.page_count) == (DocumentStatus.READY, 2)
        job = await session.scalar(select(models.IngestionJob))
        assert job is not None
        assert (job.status, job.stage, job.attempts) == (JobStatus.SUCCEEDED, "done", 1)

        chunks = (
            await session.scalars(select(models.Chunk).order_by(models.Chunk.chunk_index))
        ).all()
        # The decorative "logo" figure is dropped; text, described figure and table remain.
        assert [c.modality for c in chunks] == [
            ChunkModality.TEXT,
            ChunkModality.FIGURE,
            ChunkModality.TABLE,
        ]
        text_chunk, figure, table = chunks
        assert text_chunk.section_path == "Installation"
        assert figure.content == "Figure 1: Wiring\n\nA diagram in Installation."
        assert table.section_path == "Alarms"
        assert all(c.embedding is not None and c.embedding_model == "fake-embedder" for c in chunks)
        assert all(c.collection_id == document.collection_id for c in chunks)
        assert text_chunk.bbox == {"x0": 10, "y0": 20, "x1": 200, "y1": 60}
        assert text_chunk.meta["regions"][0]["page"] == 1

        # Section headings are searchable through the full-text index.
        hit = await session.scalar(
            select(models.Chunk.chunk_index).where(
                models.Chunk.search_vector.op("@@")(func.plainto_tsquery("english", "alarms"))
            )
        )
        assert hit == table.chunk_index

        assert await storage.get(page_key(document_id, 2)) == PNG
        assert figure.asset_key is not None
        assert await storage.get(figure.asset_key) == PNG
        assert text_chunk.asset_key is None


async def test_reingesting_replaces_rather_than_duplicates(
    session_factory: Factory, storage: LocalStorage
) -> None:
    document_id = await upload(session_factory, storage)
    pipeline = make_pipeline(storage)

    await pipeline.run(session_factory, document_id)
    await pipeline.run(session_factory, document_id)

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(models.Chunk)) == 3
        assert await session.scalar(select(func.count()).select_from(models.DocumentPage)) == 2


async def test_failed_job_is_retried_then_succeeds(
    session_factory: Factory, storage: LocalStorage, settings: Settings
) -> None:
    document_id = await upload(session_factory, storage)
    pipeline = make_pipeline(storage, FakeParser(fail_times=1))

    await run_next_job(session_factory, pipeline, settings)
    async with session_factory() as session:
        job = await session.scalar(select(models.IngestionJob))
        assert job is not None
        assert (job.status, job.attempts) == (JobStatus.QUEUED, 1)
        assert job.error == "RuntimeError: parser exploded"

    await run_next_job(session_factory, pipeline, settings)
    async with session_factory() as session:
        document = await session.get_one(models.Document, document_id)
        assert document.status is DocumentStatus.READY


async def test_job_fails_permanently_after_max_attempts(
    session_factory: Factory, storage: LocalStorage, settings: Settings
) -> None:
    document_id = await upload(session_factory, storage)
    pipeline = make_pipeline(storage, FakeParser(fail_times=99))

    for _ in range(settings.ingest_max_attempts):
        assert await run_next_job(session_factory, pipeline, settings) is True
    assert await run_next_job(session_factory, pipeline, settings) is False

    async with session_factory() as session:
        document = await session.get_one(models.Document, document_id)
        assert document.status is DocumentStatus.FAILED
        assert document.error == "RuntimeError: parser exploded"
        assert await session.scalar(select(func.count()).select_from(models.Chunk)) == 0


async def test_concurrent_workers_never_claim_the_same_job(
    session_factory: Factory, storage: LocalStorage
) -> None:
    for i in range(3):
        await upload(session_factory, storage, PDF + bytes([i]))

    async def claim() -> uuid.UUID | None:
        async with session_factory() as session:
            job = await jobs.claim_job(session, stale_after=STALE, max_attempts=3)
            return job.id if job else None

    claimed = await asyncio.gather(*(claim() for _ in range(8)))
    taken = [job_id for job_id in claimed if job_id is not None]
    assert len(taken) == 3
    assert len(set(taken)) == 3


async def test_job_abandoned_by_a_dead_worker_is_reclaimed(
    session_factory: Factory, storage: LocalStorage
) -> None:
    await upload(session_factory, storage)
    async with session_factory() as session:
        first = await jobs.claim_job(session, stale_after=STALE, max_attempts=3)
        assert first is not None
        # Still fresh: nobody else may take it.
        assert await jobs.claim_job(session, stale_after=STALE, max_attempts=3) is None

        await session.execute(
            update(models.IngestionJob).values(started_at=datetime.now(UTC) - timedelta(hours=1))
        )
        await session.commit()
        again = await jobs.claim_job(session, stale_after=STALE, max_attempts=3)
        assert again is not None
        assert (again.id, again.attempts) == (first.id, 2)


async def test_job_abandoned_on_its_last_attempt_is_failed(
    session_factory: Factory, storage: LocalStorage
) -> None:
    document_id = await upload(session_factory, storage)
    async with session_factory() as session:
        assert await jobs.claim_job(session, stale_after=STALE, max_attempts=1) is not None
        await session.execute(
            update(models.IngestionJob).values(started_at=datetime.now(UTC) - timedelta(hours=1))
        )
        await session.commit()

        assert await jobs.claim_job(session, stale_after=STALE, max_attempts=1) is None
        document = await session.get_one(models.Document, document_id)
        await session.refresh(document)
        assert document.status is DocumentStatus.FAILED


async def test_document_with_no_extractable_content_fails_clearly(
    session_factory: Factory, storage: LocalStorage
) -> None:
    from intelliquery.ingestion.types import ParsedDocument

    document_id = await upload(session_factory, storage)
    pipeline = make_pipeline(storage, FakeParser(ParsedDocument()))
    with pytest.raises(ValueError, match="no extractable content"):
        await pipeline.run(session_factory, document_id)
