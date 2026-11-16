"""Ingestion worker: claims queued jobs and runs the pipeline."""

import asyncio
import contextlib
import logging
import signal
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelliquery.config import Settings
from intelliquery.ingestion import jobs
from intelliquery.ingestion.pipeline import IngestionPipeline
from intelliquery.providers.openai_compat import (
    OpenAICompatClient,
    OpenAICompatEmbedder,
    OpenAICompatFigureDescriber,
)
from intelliquery.storage import LocalStorage

logger = logging.getLogger(__name__)


def build_pipeline(settings: Settings) -> tuple[IngestionPipeline, list[OpenAICompatClient]]:
    # Deferred: importing the parser loads PyTorch.
    from intelliquery.ingestion.parsers.docling_parser import DoclingParser

    def client(base_url: str, key: str | None) -> OpenAICompatClient:
        return OpenAICompatClient(
            base_url,
            key,
            timeout=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
        )

    embedding_key = settings.embedding_api_key
    embedding_client = client(
        settings.embedding_base_url, embedding_key.get_secret_value() if embedding_key else None
    )
    clients = [embedding_client]
    describer = None
    if settings.vision_model:
        vision_key = settings.vision_api_key
        vision_client = client(
            settings.vision_base_url, vision_key.get_secret_value() if vision_key else None
        )
        clients.append(vision_client)
        describer = OpenAICompatFigureDescriber(vision_client, settings.vision_model)
    else:
        logger.warning("no vision model configured; figures are indexed by caption only")

    pipeline = IngestionPipeline(
        parser=DoclingParser(image_scale=settings.ingest_image_scale, ocr=settings.ingest_ocr),
        embedder=OpenAICompatEmbedder(
            embedding_client,
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            request_dimensions=settings.embedding_request_dimensions,
        ),
        describer=describer,
        storage=LocalStorage(settings.storage_dir),
        max_chunk_tokens=settings.ingest_max_chunk_tokens,
        vision_concurrency=settings.vision_concurrency,
    )
    return pipeline, clients


async def run_next_job(
    session_factory: async_sessionmaker[AsyncSession],
    pipeline: IngestionPipeline,
    settings: Settings,
) -> bool:
    """Process one job. Returns False when the queue was empty."""
    async with session_factory() as session:
        job = await jobs.claim_job(
            session,
            stale_after=timedelta(seconds=settings.ingest_job_timeout_seconds),
            max_attempts=settings.ingest_max_attempts,
        )
    if job is None:
        return False
    job_id, document_id = job.id, job.document_id
    log_context = {"job_id": str(job_id), "document_id": str(document_id)}

    async def on_stage(stage: str) -> None:
        async with session_factory() as session:
            await jobs.set_stage(session, job_id, stage)

    try:
        await pipeline.run(session_factory, document_id, on_stage)
    except Exception as exc:
        logger.exception("ingestion failed", extra=log_context)
        async with session_factory() as session:
            retrying = await jobs.fail_job(
                session,
                job_id,
                f"{type(exc).__name__}: {exc}",
                max_attempts=settings.ingest_max_attempts,
            )
        logger.info(
            "job will be retried" if retrying else "job failed permanently", extra=log_context
        )
    else:
        async with session_factory() as session:
            await jobs.complete_job(session, job_id)
    return True


async def run_worker(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    drain: bool = False,
) -> None:
    """Run until stopped. With `drain`, exit as soon as the queue is empty."""
    pipeline, clients = build_pipeline(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:  # Windows event loops
            signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stop.set))

    logger.info("worker started")
    try:
        while not stop.is_set():
            worked = await run_next_job(session_factory, pipeline, settings)
            if worked:
                continue
            if drain:
                break
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_seconds)
    finally:
        for model_client in clients:
            await model_client.aclose()
        logger.info("worker stopped")
