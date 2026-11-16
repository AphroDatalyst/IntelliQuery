"""Request and response bodies of the HTTP API."""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from intelliquery.db.models import ChunkModality, DocumentStatus, MessageRole

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class _FromRow(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Info(BaseModel):
    version: str
    # Null when no chat model is configured: documents can be managed, questions cannot be asked.
    chat_model: str | None
    embedding_model: str
    max_upload_mb: int


class CollectionCreate(BaseModel):
    name: Name
    description: str | None = Field(default=None, max_length=2000)


class CollectionOut(_FromRow):
    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    document_count: int = 0


class DocumentOut(_FromRow):
    id: uuid.UUID
    collection_id: uuid.UUID
    filename: str
    status: DocumentStatus
    size_bytes: int
    page_count: int | None
    error: str | None
    created_at: datetime
    chunk_count: int = 0


class Box(BaseModel):
    """A rectangle in page-image pixels."""

    x0: float
    y0: float
    x1: float
    y1: float


class Region(BaseModel):
    """Where on a page a cited chunk sits, with the page size needed to scale the box."""

    page: int
    box: Box | None
    page_width: int | None
    page_height: int | None


class CitationOut(BaseModel):
    marker: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    modality: ChunkModality
    page_number: int | None
    section_path: str | None
    content: str
    has_image: bool
    regions: list[Region]
    # True or false once the verification pass judged the citation; null if it did not run.
    supported: bool | None


class MessageOut(BaseModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    created_at: datetime
    abstained: bool = False
    # Every citation of the answer passed verification.
    verified: bool = False
    model: str | None = None
    latency_ms: int | None = None
    citations: list[CitationOut] = []
    rating: Literal[-1, 1] | None = None


class ConversationOut(_FromRow):
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str | None
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class AskRequest(BaseModel):
    question: Question
    # Omit to start a new conversation.
    conversation_id: uuid.UUID | None = None


class AskResponse(BaseModel):
    conversation: ConversationOut
    question: MessageOut
    answer: MessageOut
    # Sentences removed for lacking a valid citation or failing verification.
    dropped_sentences: int


class FeedbackIn(BaseModel):
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=2000)
