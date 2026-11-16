"""Request dependencies: database session, acting user, storage, answer service."""

import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.answering.service import (
    AnswerService,
    ChatModelNotConfigured,
    build_answer_service,
)
from intelliquery.config import Settings
from intelliquery.db import models
from intelliquery.identity import get_local_user
from intelliquery.storage import ObjectStorage


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """One transaction per request: committed when the handler returns, else rolled back."""
    async with request.app.state.session_factory() as session, session.begin():
        yield session


def get_storage(request: Request) -> ObjectStorage:
    storage: ObjectStorage = request.app.state.storage
    return storage


def get_answer_service(request: Request) -> AnswerService:
    """The answer service, built on first use so the API starts without a chat model."""
    state = request.app.state
    if state.answer_service is None:
        try:
            state.answer_service, clients = build_answer_service(state.settings)
        except ChatModelNotConfigured as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        state.model_clients.extend(clients)
    service: AnswerService = state.answer_service
    return service


SettingsDep = Annotated[Settings, Depends(get_settings)]
# "function" scope ends the transaction before the response is sent, so a client that
# reads straight after a write always sees it.
SessionDep = Annotated[AsyncSession, Depends(get_session, scope="function")]
StorageDep = Annotated[ObjectStorage, Depends(get_storage)]
AnswerServiceDep = Annotated[AnswerService, Depends(get_answer_service)]


async def get_current_user(session: SessionDep) -> models.User:
    return await get_local_user(session)


UserDep = Annotated[models.User, Depends(get_current_user)]


def not_found(what: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"{what} not found.")


# Every lookup below is scoped to the acting user. Something that exists but belongs
# to someone else is reported as missing, so ids cannot be probed.


async def owned_collection(
    session: AsyncSession, user: models.User, collection_id: uuid.UUID
) -> models.Collection:
    collection = await session.scalar(
        select(models.Collection).where(
            models.Collection.id == collection_id, models.Collection.owner_id == user.id
        )
    )
    if collection is None:
        raise not_found("Collection")
    return collection


async def owned_document(
    session: AsyncSession, user: models.User, document_id: uuid.UUID
) -> models.Document:
    document = await session.scalar(
        select(models.Document)
        .join(models.Collection, models.Collection.id == models.Document.collection_id)
        .where(models.Document.id == document_id, models.Collection.owner_id == user.id)
    )
    if document is None:
        raise not_found("Document")
    return document


async def owned_conversation(
    session: AsyncSession, user: models.User, conversation_id: uuid.UUID
) -> models.Conversation:
    conversation = await session.scalar(
        select(models.Conversation).where(
            models.Conversation.id == conversation_id, models.Conversation.user_id == user.id
        )
    )
    if conversation is None:
        raise not_found("Conversation")
    return conversation
