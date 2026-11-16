"""Persisting conversations, answers and their citations."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.answering.service import Answer, History
from intelliquery.db import models
from intelliquery.db.models import MessageRole

TITLE_LENGTH = 80


async def load_history(session: AsyncSession, conversation_id: uuid.UUID, turns: int) -> History:
    """Return the last `turns` question/answer pairs, oldest first."""
    if turns <= 0:
        return []
    rows = await session.execute(
        select(models.Message.role, models.Message.content)
        .where(models.Message.conversation_id == conversation_id)
        # Both messages of a turn share a transaction timestamp; role breaks the tie.
        .order_by(models.Message.created_at.desc(), models.Message.role.desc())
        .limit(turns * 2)
    )
    return [(role.value, content) for role, content in reversed(rows.all())]


async def record_turn(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    question: str,
    answer: Answer,
    conversation_id: uuid.UUID | None = None,
) -> models.Message:
    """Store the question and the answer; start a conversation if none is given."""
    if conversation_id is None:
        conversation = models.Conversation(
            user_id=user_id, collection_id=collection_id, title=question[:TITLE_LENGTH]
        )
        session.add(conversation)
        await session.flush()
        conversation_id = conversation.id

    session.add(
        models.Message(conversation_id=conversation_id, role=MessageRole.USER, content=question)
    )
    message = models.Message(
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content=answer.text,
        abstained=answer.abstained,
        model=answer.model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        latency_ms=answer.latency_ms,
        citations=[
            models.Citation(
                chunk_id=citation.chunk.chunk_id,
                marker=citation.marker,
                support_score=None if citation.supported is None else float(citation.supported),
            )
            for citation in answer.citations
        ],
    )
    session.add(message)
    await session.flush()
    return message
