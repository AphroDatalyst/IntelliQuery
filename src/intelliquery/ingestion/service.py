"""Accepting documents: validate, deduplicate, store, enqueue."""

import hashlib
import uuid
from pathlib import PurePath

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.db import models
from intelliquery.storage import ObjectStorage

PDF_MAGIC = b"%PDF-"
PDF_CONTENT_TYPE = "application/pdf"


class UploadRejected(ValueError):
    """The file cannot be ingested; the message is safe to show to the user."""


def source_key(document_id: uuid.UUID) -> str:
    return f"documents/{document_id}/source.pdf"


def page_key(document_id: uuid.UUID, page_number: int) -> str:
    return f"documents/{document_id}/pages/{page_number}.png"


def asset_key(document_id: uuid.UUID, chunk_index: int) -> str:
    return f"documents/{document_id}/assets/{chunk_index}.png"


def document_prefix(document_id: uuid.UUID) -> str:
    return f"documents/{document_id}"


def validate_upload(filename: str, data: bytes, *, max_bytes: int) -> str:
    """Return a safe display name, or raise UploadRejected."""
    if not data:
        raise UploadRejected("The file is empty.")
    if len(data) > max_bytes:
        raise UploadRejected(f"The file is larger than {max_bytes // (1024 * 1024)} MB.")
    # Trust the content, not the extension or the client-supplied type.
    if not data.startswith(PDF_MAGIC):
        raise UploadRejected("Only PDF files are supported.")
    # Drop any client-supplied directory components.
    name = PurePath(filename.replace("\\", "/")).name.strip()
    return name[:512] or "document.pdf"


async def register_document(
    session: AsyncSession,
    storage: ObjectStorage,
    *,
    collection_id: uuid.UUID,
    filename: str,
    data: bytes,
    max_bytes: int,
) -> tuple[models.Document, bool]:
    """Store a file and queue it for ingestion.

    Returns the document and whether it was newly created. Uploading the same bytes
    to the same collection again returns the existing document.
    """
    name = validate_upload(filename, data, max_bytes=max_bytes)
    digest = hashlib.sha256(data).hexdigest()

    existing = await session.scalar(
        select(models.Document).where(
            models.Document.collection_id == collection_id, models.Document.sha256 == digest
        )
    )
    if existing is not None:
        return existing, False

    document_id = uuid.uuid4()
    await storage.put(source_key(document_id), data)
    document = models.Document(
        id=document_id,
        collection_id=collection_id,
        filename=name,
        content_type=PDF_CONTENT_TYPE,
        sha256=digest,
        size_bytes=len(data),
        storage_key=source_key(document_id),
    )
    session.add(document)
    session.add(models.IngestionJob(document_id=document_id))
    await session.flush()
    return document, True
