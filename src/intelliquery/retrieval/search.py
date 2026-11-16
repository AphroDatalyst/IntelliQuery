"""Hybrid search over one collection."""

import logging
import uuid
from collections.abc import Sequence

from sqlalchemy import Text, func, literal, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.db import models
from intelliquery.providers import Embedder, Reranker
from intelliquery.retrieval import RetrievedChunk
from intelliquery.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# ts_rank_cd normalisation: divide by (1 + log(document length)), so long chunks do
# not win merely by containing more words.
_RANK_NORMALISATION = 1


async def dense_search(
    session: AsyncSession, collection_id: uuid.UUID, vector: Sequence[float], limit: int
) -> list[uuid.UUID]:
    # With a collection filter, a plain HNSW scan can return fewer than `limit` rows.
    # Iterative scanning keeps walking the graph until enough rows pass the filter.
    await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    await session.execute(text("SET LOCAL hnsw.ef_search = 100"))
    rows = await session.scalars(
        select(models.Chunk.id)
        .where(models.Chunk.collection_id == collection_id, models.Chunk.embedding.is_not(None))
        .order_by(models.Chunk.embedding.cosine_distance(list(vector)))
        .limit(limit)
    )
    return list(rows)


async def keyword_search(
    session: AsyncSession, collection_id: uuid.UUID, query: str, limit: int
) -> list[uuid.UUID]:
    # A question is not a boolean query: requiring every word (AND) returns nothing
    # for most natural-language questions. Match any term and let ranking sort it out.
    lexemes = func.plainto_tsquery("english", query).cast(Text)
    ts_query = func.to_tsquery("english", func.replace(lexemes, literal("&"), literal("|")))
    rank = func.ts_rank_cd(models.Chunk.search_vector, ts_query, _RANK_NORMALISATION)
    rows = await session.scalars(
        select(models.Chunk.id)
        .where(
            models.Chunk.collection_id == collection_id,
            # Stop-word-only questions produce an empty query, which matches nothing.
            func.numnode(ts_query) > 0,
            models.Chunk.search_vector.op("@@")(ts_query),
        )
        .order_by(rank.desc(), models.Chunk.id)
        .limit(limit)
    )
    return list(rows)


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        candidates: int = 30,
        top_k: int = 8,
        rerank_min_score: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._candidates = candidates
        self._top_k = top_k
        self._rerank_min_score = rerank_min_score

    async def retrieve(
        self, session: AsyncSession, collection_id: uuid.UUID, query: str
    ) -> list[RetrievedChunk]:
        (vector,) = await self._embedder.embed([query])
        dense = await dense_search(session, collection_id, vector, self._candidates)
        keyword = await keyword_search(session, collection_id, query, self._candidates)
        fused = reciprocal_rank_fusion([dense, keyword])
        if not fused:
            return []

        # Without a reranker the fused order is final, so only the top results are loaded.
        shortlist = fused[: self._candidates if self._reranker else self._top_k]
        rows = await session.execute(
            select(models.Chunk, models.Document.filename)
            .join(models.Document, models.Document.id == models.Chunk.document_id)
            .where(models.Chunk.id.in_([chunk_id for chunk_id, _ in shortlist]))
        )
        loaded = {chunk.id: (chunk, filename) for chunk, filename in rows}
        dense_rank = {chunk_id: i for i, chunk_id in enumerate(dense, start=1)}
        keyword_rank = {chunk_id: i for i, chunk_id in enumerate(keyword, start=1)}

        results = []
        for chunk_id, score in shortlist:
            if chunk_id not in loaded:  # deleted between the two queries
                continue
            chunk, filename = loaded[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    filename=filename,
                    modality=chunk.modality,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    section_path=chunk.section_path,
                    asset_key=chunk.asset_key,
                    bbox=chunk.bbox,
                    regions=chunk.meta.get("regions", []),
                    dense_rank=dense_rank.get(chunk_id),
                    keyword_rank=keyword_rank.get(chunk_id),
                    fused_score=score,
                )
            )

        if self._reranker and results:
            scores = await self._reranker.rerank(
                query, [_rerank_text(result) for result in results]
            )
            for result, rerank_score in zip(results, scores, strict=True):
                result.rerank_score = rerank_score
            results = [r for r in results if (r.rerank_score or 0.0) >= self._rerank_min_score]
            results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)

        logger.info(
            "retrieved",
            extra={"dense": len(dense), "keyword": len(keyword), "returned": len(results)},
        )
        return results[: self._top_k]


def _rerank_text(chunk: RetrievedChunk) -> str:
    return f"{chunk.section_path}\n\n{chunk.content}" if chunk.section_path else chunk.content
