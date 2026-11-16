"""Collections and the documents in them."""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, UploadFile, status
from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError

from intelliquery import __version__
from intelliquery.api.deps import (
    SessionDep,
    SettingsDep,
    StorageDep,
    UserDep,
    not_found,
    owned_collection,
    owned_document,
)
from intelliquery.api.schemas import CollectionCreate, CollectionOut, DocumentOut, Info
from intelliquery.db import models
from intelliquery.ingestion.service import UploadRejected, document_prefix, register_document
from intelliquery.storage import ObjectStorage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])

_READ_CHUNK = 1024 * 1024
_PNG = "image/png"
# Stored images never change for a given document, so the browser may keep them.
_IMAGE_CACHE = "private, max-age=86400, immutable"


@router.get("/info")
async def info(settings: SettingsDep) -> Info:
    """What this deployment can do, for the UI."""
    return Info(
        version=__version__,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        max_upload_mb=settings.ingest_max_upload_mb,
    )


@router.get("/collections")
async def list_collections(session: SessionDep, user: UserDep) -> list[CollectionOut]:
    counts = (
        select(models.Document.collection_id, func.count().label("documents"))
        .group_by(models.Document.collection_id)
        .subquery()
    )
    rows = await session.execute(
        select(models.Collection, func.coalesce(counts.c.documents, 0))
        .outerjoin(counts, counts.c.collection_id == models.Collection.id)
        .where(models.Collection.owner_id == user.id)
        .order_by(models.Collection.created_at)
    )
    return [
        CollectionOut.model_validate(collection).model_copy(update={"document_count": count})
        for collection, count in rows
    ]


@router.post("/collections", status_code=status.HTTP_201_CREATED)
async def create_collection(
    body: CollectionCreate, session: SessionDep, user: UserDep
) -> CollectionOut:
    collection = models.Collection(owner_id=user.id, name=body.name, description=body.description)
    session.add(collection)
    try:
        await session.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A collection named {body.name!r} already exists."
        ) from exc
    return CollectionOut.model_validate(collection)


async def _delete_stored_files(storage: ObjectStorage, document_ids: list[uuid.UUID]) -> None:
    for document_id in document_ids:
        try:
            await storage.delete_prefix(document_prefix(document_id))
        except OSError:
            # The database rows are gone; orphaned files cost disk space, not correctness.
            logger.exception("could not delete stored files", extra={"document_id": document_id})


@router.delete("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: uuid.UUID,
    session: SessionDep,
    user: UserDep,
    storage: StorageDep,
    background: BackgroundTasks,
) -> Response:
    """Delete a collection with its documents, conversations and everything derived."""
    collection = await owned_collection(session, user, collection_id)
    document_ids = list(
        await session.scalars(
            select(models.Document.id).where(models.Document.collection_id == collection.id)
        )
    )
    await session.delete(collection)
    # Files are removed after the transaction commits, so a failed delete loses nothing.
    background.add_task(_delete_stored_files, storage, document_ids)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _documents_query(
    collection_id: uuid.UUID | None = None, document_id: uuid.UUID | None = None
) -> Select[models.Document, Any]:
    counts = (
        select(models.Chunk.document_id, func.count().label("chunks"))
        .group_by(models.Chunk.document_id)
        .subquery()
    )
    query = select(models.Document, func.coalesce(counts.c.chunks, 0)).outerjoin(
        counts, counts.c.document_id == models.Document.id
    )
    if collection_id is not None:
        query = query.where(models.Document.collection_id == collection_id)
    if document_id is not None:
        query = query.where(models.Document.id == document_id)
    return query.order_by(models.Document.created_at, models.Document.id)


def _document_out(document: models.Document, chunks: int) -> DocumentOut:
    return DocumentOut.model_validate(document).model_copy(update={"chunk_count": chunks})


@router.get("/collections/{collection_id}/documents")
async def list_documents(
    collection_id: uuid.UUID, session: SessionDep, user: UserDep
) -> list[DocumentOut]:
    collection = await owned_collection(session, user, collection_id)
    rows = await session.execute(_documents_query(collection_id=collection.id))
    return [_document_out(document, chunks) for document, chunks in rows]


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, giving up as soon as it exceeds the limit."""
    parts: list[bytes] = []
    total = 0
    while chunk := await file.read(_READ_CHUNK):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"The file is larger than {max_bytes // (1024 * 1024)} MB.",
            )
        parts.append(chunk)
    return b"".join(parts)


@router.post(
    "/collections/{collection_id}/documents",
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "The same file was already in the collection."}},
)
async def upload_document(
    collection_id: uuid.UUID,
    file: UploadFile,
    response: Response,
    session: SessionDep,
    user: UserDep,
    storage: StorageDep,
    settings: SettingsDep,
) -> DocumentOut:
    """Store a PDF and queue it for ingestion. Re-uploading the same file is a no-op."""
    collection = await owned_collection(session, user, collection_id)
    max_bytes = settings.ingest_max_upload_mb * 1024 * 1024
    data = await _read_limited(file, max_bytes)
    try:
        document, created = await register_document(
            session,
            storage,
            collection_id=collection.id,
            filename=file.filename or "document.pdf",
            data=data,
            max_bytes=max_bytes,
        )
    except UploadRejected as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    if not created:
        response.status_code = status.HTTP_200_OK
    document, chunks = (await session.execute(_documents_query(document_id=document.id))).one()
    return _document_out(document, chunks)


@router.get("/documents/{document_id}")
async def get_document(document_id: uuid.UUID, session: SessionDep, user: UserDep) -> DocumentOut:
    await owned_document(session, user, document_id)
    document, chunks = (await session.execute(_documents_query(document_id=document_id))).one()
    return _document_out(document, chunks)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    session: SessionDep,
    user: UserDep,
    storage: StorageDep,
    background: BackgroundTasks,
) -> Response:
    """Delete a document with its pages, chunks and the citations that pointed at them."""
    document = await owned_document(session, user, document_id)
    await session.delete(document)
    background.add_task(_delete_stored_files, storage, [document.id])
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _image(storage: ObjectStorage, key: str) -> Response:
    try:
        data = await storage.get(key)
    except OSError as exc:
        raise not_found("Image") from exc
    return Response(data, media_type=_PNG, headers={"Cache-Control": _IMAGE_CACHE})


@router.get("/documents/{document_id}/pages/{page_number}/image", response_class=Response)
async def page_image(
    document_id: uuid.UUID,
    page_number: int,
    session: SessionDep,
    user: UserDep,
    storage: StorageDep,
) -> Response:
    """The rendered page, on which the UI draws the cited region."""
    document = await owned_document(session, user, document_id)
    key = await session.scalar(
        select(models.DocumentPage.image_key).where(
            models.DocumentPage.document_id == document.id,
            models.DocumentPage.page_number == page_number,
        )
    )
    if key is None:
        raise not_found("Page")
    return await _image(storage, key)


@router.get("/chunks/{chunk_id}/image", response_class=Response)
async def chunk_image(
    chunk_id: uuid.UUID, session: SessionDep, user: UserDep, storage: StorageDep
) -> Response:
    """The cropped image of a figure or table chunk."""
    key = await session.scalar(
        select(models.Chunk.asset_key)
        .join(models.Collection, models.Collection.id == models.Chunk.collection_id)
        .where(models.Chunk.id == chunk_id, models.Collection.owner_id == user.id)
    )
    if key is None:
        raise not_found("Image")
    return await _image(storage, key)
