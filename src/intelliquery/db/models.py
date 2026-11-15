"""Relational schema.

One PostgreSQL database holds everything the system needs to answer a question and
prove where the answer came from: source documents, their pages, retrievable chunks
(dense vector + full-text index), conversations, and per-claim citations.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from intelliquery.config import EMBEDDING_DIMENSIONS
from intelliquery.db.base import Base


def _enum(enum_cls: type[enum.Enum], name: str) -> Enum:
    return Enum(enum_cls, name=name, values_callable=lambda e: [m.value for m in e])


class UserRole(enum.StrEnum):
    ADMIN = "admin"
    MEMBER = "member"


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ChunkModality(enum.StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class MessageRole(enum.StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class _IdMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


class _CreatedMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class User(_IdMixin, _CreatedMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        _enum(UserRole, "user_role"), server_default=UserRole.MEMBER.value, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"), nullable=False)

    collections: Mapped[list[Collection]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )


class Collection(_IdMixin, _CreatedMixin, Base):
    """A named set of documents; retrieval is always scoped to one collection."""

    __tablename__ = "collections"
    __table_args__ = (UniqueConstraint("owner_id", "name"),)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    owner: Mapped[User] = relationship(back_populates="collections")
    documents: Mapped[list[Document]] = relationship(
        back_populates="collection", cascade="all, delete-orphan", passive_deletes=True
    )


class Document(_IdMixin, _CreatedMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        # The same file uploaded twice to a collection is a no-op, not a duplicate index.
        UniqueConstraint("collection_id", "sha256"),
    )

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(127), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[DocumentStatus] = mapped_column(
        _enum(DocumentStatus, "document_status"),
        server_default=DocumentStatus.PENDING.value,
        index=True,
        nullable=False,
    )
    error: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    collection: Mapped[Collection] = relationship(back_populates="documents")
    pages: Mapped[list[DocumentPage]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentPage(_IdMixin, Base):
    """A rendered page image, used to show the cited region to the reader."""

    __tablename__ = "document_pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number"),)

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    image_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    document: Mapped[Document] = relationship(back_populates="pages")


class IngestionJob(_IdMixin, _CreatedMixin, Base):
    __tablename__ = "ingestion_jobs"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        _enum(JobStatus, "job_status"),
        server_default=JobStatus.QUEUED.value,
        index=True,
        nullable=False,
    )
    stage: Mapped[str | None] = mapped_column(String(64))
    attempts: Mapped[int] = mapped_column(SmallInteger, server_default=text("0"), nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Chunk(_IdMixin, _CreatedMixin, Base):
    """The unit of retrieval: a passage, a table, or a figure with its description.

    `content` is what the reader is shown and what citations quote. `context` is a short
    generated summary situating the chunk within its document; it is indexed alongside the
    content so retrieval sees it, but it is never quoted.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    # Denormalised from documents so the filtered vector search needs no join.
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True, nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    modality: Mapped[ChunkModality] = mapped_column(
        _enum(ChunkModality, "chunk_modality"), nullable=False
    )
    page_number: Mapped[int | None] = mapped_column(Integer)
    # Region on the page, in page-image pixels: {"x0":..,"y0":..,"x1":..,"y1":..}
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    section_path: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str | None] = mapped_column(Text)
    # Cropped image for figure and table chunks.
    asset_key: Mapped[str | None] = mapped_column(String(1024))
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(context, '') || ' ' || content)", persisted=True),
    )
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb"), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(_IdMixin, _CreatedMixin, Base):
    __tablename__ = "conversations"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str | None] = mapped_column(String(300))

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(_IdMixin, _CreatedMixin, Base):
    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(_enum(MessageRole, "message_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # True when the assistant declined because the documents did not support an answer.
    abstained: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
    model: Mapped[str | None] = mapped_column(String(200))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    trace_id: Mapped[str | None] = mapped_column(String(64))

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    citations: Mapped[list[Citation]] = relationship(
        back_populates="message", cascade="all, delete-orphan", passive_deletes=True
    )


class Citation(_IdMixin, Base):
    """Links one claim in an assistant message to the chunk that supports it."""

    __tablename__ = "citations"
    __table_args__ = (UniqueConstraint("message_id", "marker"),)

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    # Chunks cannot be deleted from under a citation without deleting the document,
    # which cascades here through chunks.
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), index=True, nullable=False
    )
    marker: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    quote: Mapped[str | None] = mapped_column(Text)
    # Entailment score from the verification pass; null until verified.
    support_score: Mapped[float | None] = mapped_column(Float)

    message: Mapped[Message] = relationship(back_populates="citations")
    chunk: Mapped[Chunk] = relationship()


class Feedback(_IdMixin, _CreatedMixin, Base):
    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("message_id", "user_id"),
        CheckConstraint("rating IN (-1, 1)", name="rating_is_vote"),
    )

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)


class EvalRun(_IdMixin, _CreatedMixin, Base):
    """One offline evaluation of the pipeline, so quality can be tracked per commit."""

    __tablename__ = "eval_runs"

    dataset: Mapped[str] = mapped_column(String(200), index=True, nullable=False)
    git_sha: Mapped[str | None] = mapped_column(String(40))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
