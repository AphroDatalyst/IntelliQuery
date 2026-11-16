"""Asking questions, and the conversations that result."""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.answering.conversations import load_history, record_turn
from intelliquery.api.deps import (
    AnswerServiceDep,
    SessionDep,
    SettingsDep,
    UserDep,
    not_found,
    owned_collection,
    owned_conversation,
)
from intelliquery.api.schemas import (
    AskRequest,
    AskResponse,
    ConversationDetail,
    ConversationOut,
    FeedbackIn,
)
from intelliquery.api.views import conversation_messages, message_views
from intelliquery.db import models
from intelliquery.db.models import MessageRole
from intelliquery.providers import ProviderError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["conversations"])


@router.post("/collections/{collection_id}/ask")
async def ask(
    collection_id: uuid.UUID,
    body: AskRequest,
    session: SessionDep,
    user: UserDep,
    settings: SettingsDep,
    service: AnswerServiceDep,
) -> AskResponse:
    """Answer a question from the collection's documents and store the exchange.

    The answer contains only sentences tied to a source; when the documents do not
    support an answer, the reply says so and is marked `abstained`.
    """
    collection = await owned_collection(session, user, collection_id)
    history = []
    if body.conversation_id is not None:
        conversation = await owned_conversation(session, user, body.conversation_id)
        if conversation.collection_id != collection.id:
            raise not_found("Conversation")
        history = list(await load_history(session, conversation.id, settings.answer_history_turns))

    try:
        answer = await service.answer(session, collection.id, body.question, history)
    except ProviderError as exc:
        logger.exception("model provider failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "A model endpoint failed. Please try again."
        ) from exc

    message = await record_turn(
        session,
        user_id=user.id,
        collection_id=collection.id,
        question=body.question,
        answer=answer,
        conversation_id=body.conversation_id,
    )
    conversation = await session.get_one(models.Conversation, message.conversation_id)
    turn = await conversation_messages(session, conversation.id, last=2)
    question_view, answer_view = await message_views(session, user.id, turn)
    return AskResponse(
        conversation=ConversationOut.model_validate(conversation),
        question=question_view,
        answer=answer_view,
        dropped_sentences=answer.dropped_sentences,
    )


@router.get("/collections/{collection_id}/conversations")
async def list_conversations(
    collection_id: uuid.UUID, session: SessionDep, user: UserDep
) -> list[ConversationOut]:
    """Conversations in a collection, newest first."""
    collection = await owned_collection(session, user, collection_id)
    rows = await session.scalars(
        select(models.Conversation)
        .where(
            models.Conversation.collection_id == collection.id,
            models.Conversation.user_id == user.id,
        )
        .order_by(models.Conversation.created_at.desc(), models.Conversation.id)
    )
    return [ConversationOut.model_validate(row) for row in rows]


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: uuid.UUID, session: SessionDep, user: UserDep
) -> ConversationDetail:
    conversation = await owned_conversation(session, user, conversation_id)
    messages = await conversation_messages(session, conversation.id)
    return ConversationDetail(
        **ConversationOut.model_validate(conversation).model_dump(),
        messages=await message_views(session, user.id, messages),
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID, session: SessionDep, user: UserDep
) -> Response:
    await session.delete(await owned_conversation(session, user, conversation_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _owned_answer(
    session: AsyncSession, user: models.User, message_id: uuid.UUID
) -> models.Message:
    message = await session.scalar(
        select(models.Message)
        .join(models.Conversation, models.Conversation.id == models.Message.conversation_id)
        .where(
            models.Message.id == message_id,
            models.Message.role == MessageRole.ASSISTANT,
            models.Conversation.user_id == user.id,
        )
    )
    if message is None:
        raise not_found("Answer")
    return message


@router.put("/messages/{message_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def rate_answer(
    message_id: uuid.UUID, body: FeedbackIn, session: SessionDep, user: UserDep
) -> Response:
    """Rate an answer up or down; rating again replaces the earlier vote."""
    message = await _owned_answer(session, user, message_id)
    values = {"rating": body.rating, "comment": body.comment}
    await session.execute(
        insert(models.Feedback)
        .values(message_id=message.id, user_id=user.id, **values)
        .on_conflict_do_update(
            index_elements=[models.Feedback.message_id, models.Feedback.user_id], set_=values
        )
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/messages/{message_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def clear_rating(message_id: uuid.UUID, session: SessionDep, user: UserDep) -> Response:
    message = await _owned_answer(session, user, message_id)
    feedback = await session.scalar(
        select(models.Feedback).where(
            models.Feedback.message_id == message.id, models.Feedback.user_id == user.id
        )
    )
    if feedback is not None:
        await session.delete(feedback)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
