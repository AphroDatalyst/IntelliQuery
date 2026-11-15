"""A job queue on PostgreSQL.

Workers claim rows with `FOR UPDATE SKIP LOCKED`, so any number of them can poll the
same table without ever taking the same job, and the queue shares the database's
transactions and backups instead of needing a separate broker.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.db import models
from intelliquery.db.models import DocumentStatus, JobStatus

MAX_ERROR_LENGTH = 2000


async def claim_job(
    session: AsyncSession, *, stale_after: timedelta, max_attempts: int
) -> models.IngestionJob | None:
    """Atomically take the oldest runnable job, or return None when the queue is empty.

    A job left `running` past `stale_after` belonged to a worker that died, and is
    picked up again.
    """
    now = datetime.now(UTC)
    stale = (models.IngestionJob.status == JobStatus.RUNNING) & (
        models.IngestionJob.started_at < now - stale_after
    )
    # A job that died on its last allowed attempt can never be claimed again; fail it
    # so the document does not stay "processing" forever.
    exhausted = (
        await session.scalars(
            update(models.IngestionJob)
            .where(stale, models.IngestionJob.attempts >= max_attempts)
            .values(status=JobStatus.FAILED, error="worker died during ingestion", finished_at=now)
            .returning(models.IngestionJob.document_id)
        )
    ).all()
    if exhausted:
        await session.execute(
            update(models.Document)
            .where(models.Document.id.in_(exhausted))
            .values(status=DocumentStatus.FAILED, error="worker died during ingestion")
        )
    job = await session.scalar(
        select(models.IngestionJob)
        .where(
            or_(models.IngestionJob.status == JobStatus.QUEUED, stale),
            models.IngestionJob.attempts < max_attempts,
        )
        .order_by(models.IngestionJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if job is None:
        await session.commit()
        return None
    job.status = JobStatus.RUNNING
    job.attempts += 1
    job.started_at = now
    job.finished_at = None
    job.error = None
    job.stage = "claimed"
    await session.execute(
        update(models.Document)
        .where(models.Document.id == job.document_id)
        .values(status=DocumentStatus.PROCESSING, error=None)
    )
    await session.commit()
    return job


async def set_stage(session: AsyncSession, job_id: uuid.UUID, stage: str) -> None:
    await session.execute(
        update(models.IngestionJob).where(models.IngestionJob.id == job_id).values(stage=stage)
    )
    await session.commit()


async def complete_job(session: AsyncSession, job_id: uuid.UUID) -> None:
    await session.execute(
        update(models.IngestionJob)
        .where(models.IngestionJob.id == job_id)
        .values(status=JobStatus.SUCCEEDED, stage="done", finished_at=datetime.now(UTC))
    )
    await session.commit()


async def fail_job(
    session: AsyncSession, job_id: uuid.UUID, error: str, *, max_attempts: int
) -> bool:
    """Record a failure. Returns True when the job will be retried."""
    job = await session.get(models.IngestionJob, job_id, with_for_update=True)
    if job is None:
        return False
    error = error[:MAX_ERROR_LENGTH]
    will_retry = job.attempts < max_attempts
    job.status = JobStatus.QUEUED if will_retry else JobStatus.FAILED
    job.error = error
    job.finished_at = datetime.now(UTC)
    if not will_retry:
        await session.execute(
            update(models.Document)
            .where(models.Document.id == job.document_id)
            .values(status=DocumentStatus.FAILED, error=error)
        )
    await session.commit()
    return will_retry
