"""Command-line entry point: `intelliquery ingest`, `worker`, `status`."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelliquery.config import Settings, get_settings
from intelliquery.db import models
from intelliquery.db.session import create_engine, create_session_factory
from intelliquery.ingestion.service import UploadRejected, register_document
from intelliquery.logging import configure_logging
from intelliquery.storage import LocalStorage

# Until accounts exist, command-line ingestion is owned by one local user.
LOCAL_USER_EMAIL = "local@intelliquery.invalid"


async def get_or_create_collection(session: AsyncSession, name: str) -> models.Collection:
    user = await session.scalar(select(models.User).where(models.User.email == LOCAL_USER_EMAIL))
    if user is None:
        # "!" is not a valid password hash, so this account can never be logged in to.
        user = models.User(email=LOCAL_USER_EMAIL, hashed_password="!")  # noqa: S106
        session.add(user)
        await session.flush()
    collection = await session.scalar(
        select(models.Collection).where(
            models.Collection.owner_id == user.id, models.Collection.name == name
        )
    )
    if collection is None:
        collection = models.Collection(owner_id=user.id, name=name)
        session.add(collection)
        await session.flush()
    return collection


async def _ingest(
    factory: async_sessionmaker[AsyncSession], settings: Settings, args: argparse.Namespace
) -> int:
    storage = LocalStorage(settings.storage_dir)
    failures = 0
    for path in args.paths:
        async with factory() as session, session.begin():
            collection = await get_or_create_collection(session, args.collection)
            try:
                document, created = await register_document(
                    session,
                    storage,
                    collection_id=collection.id,
                    filename=path.name,
                    data=path.read_bytes(),
                    max_bytes=settings.ingest_max_upload_mb * 1024 * 1024,
                )
            except (UploadRejected, OSError) as exc:
                failures += 1
                print(f"rejected  {path.name}: {exc}", file=sys.stderr)
                continue
            state = "queued" if created else f"already present ({document.status.value})"
            print(f"{state:<28} {document.id}  {path.name}")
    return 1 if failures else 0


async def _status(factory: async_sessionmaker[AsyncSession]) -> int:
    async with factory() as session:
        chunk_counts = (
            select(models.Chunk.document_id, func.count().label("chunks"))
            .group_by(models.Chunk.document_id)
            .subquery()
        )
        rows = await session.execute(
            select(
                models.Document.filename,
                models.Document.status,
                models.Document.page_count,
                func.coalesce(chunk_counts.c.chunks, 0),
                models.Document.error,
            )
            .outerjoin(chunk_counts, chunk_counts.c.document_id == models.Document.id)
            .order_by(models.Document.created_at)
        )
        for filename, status, pages, chunks, error in rows:
            line = f"{status.value:<11} pages={pages or '-':<4} chunks={chunks:<5} {filename}"
            print(f"{line}  [{error}]" if error else line)
    return 0


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    engine = create_engine(settings)
    factory = create_session_factory(engine)
    try:
        if args.command == "ingest":
            return await _ingest(factory, settings, args)
        if args.command == "status":
            return await _status(factory)
        from intelliquery.worker import run_worker

        await run_worker(factory, settings, drain=args.drain)
        return 0
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="intelliquery")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="queue PDF files for ingestion")
    ingest.add_argument("paths", nargs="+", type=Path)
    ingest.add_argument("--collection", default="default")

    worker = commands.add_parser("worker", help="process queued ingestion jobs")
    worker.add_argument("--drain", action="store_true", help="exit when the queue is empty")

    commands.add_parser("status", help="list documents and their ingestion state")

    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
