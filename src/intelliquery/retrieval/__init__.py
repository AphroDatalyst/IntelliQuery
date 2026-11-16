"""Hybrid retrieval: dense and keyword search fused, then optionally reranked."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.db.models import ChunkModality


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    modality: ChunkModality
    content: str
    page_number: int | None = None
    section_path: str | None = None
    asset_key: str | None = None
    bbox: dict[str, Any] | None = None
    regions: list[dict[str, Any]] = field(default_factory=list)
    # Position in each ranking (1 = best); None when that search did not return the chunk.
    dense_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


class Retriever(Protocol):
    async def retrieve(
        self, session: AsyncSession, collection_id: uuid.UUID, query: str
    ) -> list[RetrievedChunk]: ...
