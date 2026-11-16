"""Hybrid retrieval and conversation storage against a real database."""

import uuid
from collections.abc import Sequence

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelliquery.answering.conversations import load_history, record_turn
from intelliquery.answering.service import Answer, Citation
from intelliquery.cli import get_or_create_collection
from intelliquery.config import EMBEDDING_DIMENSIONS
from intelliquery.db import models
from intelliquery.db.models import ChunkModality
from intelliquery.retrieval.search import HybridRetriever, keyword_search

pytestmark = pytest.mark.integration

Factory = async_sessionmaker[AsyncSession]


def axis(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


class StubEmbedder:
    """Embeds every query to one fixed vector, so dense ranking is fully controlled."""

    model = "stub"

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


class StubReranker:
    model = "stub-reranker"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        return [next(s for key, s in self._scores.items() if key in doc) for doc in documents]


async def seed(
    session: AsyncSession, name: str, chunks: Sequence[tuple[str, list[float]]]
) -> tuple[uuid.UUID, uuid.UUID]:
    """Create a collection with one document; returns (collection id, owner id)."""
    collection = await get_or_create_collection(session, name)
    document = models.Document(
        collection_id=collection.id,
        filename=f"{name}.pdf",
        content_type="application/pdf",
        sha256=uuid.uuid4().hex * 2,
        size_bytes=1,
        storage_key="k",
    )
    session.add(document)
    await session.flush()
    session.add_all(
        models.Chunk(
            document_id=document.id,
            collection_id=collection.id,
            chunk_index=index,
            modality=ChunkModality.TEXT,
            page_number=index + 1,
            content=content,
            embedding=vector,
            meta={"regions": [{"page": index + 1}]},
        )
        for index, (content, vector) in enumerate(chunks)
    )
    await session.flush()
    return collection.id, collection.owner_id


CORPUS = [
    ("The upper alarm limit for the cold room is 8 degrees.", axis(0)),
    ("Calibration certificates are renewed every twelve months.", axis(1)),
    ("Battery replacement requires a cross-head screwdriver.", axis(2)),
    ("Alarm notifications are delivered by SMS.", axis(3)),
]


async def test_fusion_ranks_chunks_found_by_both_searches_first(session_factory: Factory) -> None:
    async with session_factory() as session:
        collection_id, _ = await seed(session, "manual", CORPUS)
        # Dense search points at the battery chunk; the words point at the alarm chunks.
        blend = [0.0] * EMBEDDING_DIMENSIONS
        blend[2], blend[0] = 1.0, 0.9
        retriever = HybridRetriever(StubEmbedder(blend), top_k=4)

        results = await retriever.retrieve(session, collection_id, "what is the alarm limit?")

        top = results[0]
        assert top.content.startswith("The upper alarm limit")
        assert (top.dense_rank, top.keyword_rank) == (2, 1)
        assert top.filename == "manual.pdf"
        assert top.regions == [{"page": 1}]
        by_content = {r.content.split()[0]: r for r in results}
        # Found only by meaning, and only by keyword, respectively.
        assert by_content["Battery"].keyword_rank is None
        assert by_content["Alarm"].keyword_rank == 2
        assert [r.fused_score for r in results] == sorted(
            (r.fused_score for r in results), reverse=True
        )


async def test_retrieval_never_crosses_collections(session_factory: Factory) -> None:
    async with session_factory() as session:
        mine, _ = await seed(session, "mine", CORPUS[:2])
        theirs, _ = await seed(session, "theirs", [("Secret alarm limit is 99 degrees.", axis(0))])
        retriever = HybridRetriever(StubEmbedder(axis(0)), top_k=10)

        results = await retriever.retrieve(session, mine, "alarm limit")

        assert results
        assert all("Secret" not in r.content for r in results)
        assert len(await retriever.retrieve(session, theirs, "alarm limit")) == 1


async def test_keyword_search_matches_any_term_and_survives_stop_words(
    session_factory: Factory,
) -> None:
    async with session_factory() as session:
        collection_id, _ = await seed(session, "manual", CORPUS)

        # A natural question has words the chunk lacks; requiring all of them finds nothing.
        hits = await keyword_search(
            session, collection_id, "how often are calibration certificates renewed please", 10
        )
        assert len(hits) == 1
        # Nothing but stop words: no usable query, and no error.
        assert await keyword_search(session, collection_id, "what is the of and", 10) == []
        assert await keyword_search(session, collection_id, "", 10) == []


async def test_top_k_limits_results_and_empty_collection_returns_nothing(
    session_factory: Factory,
) -> None:
    async with session_factory() as session:
        collection_id, _ = await seed(session, "manual", CORPUS)
        empty = await get_or_create_collection(session, "empty")

        results = await HybridRetriever(StubEmbedder(axis(0)), top_k=2).retrieve(
            session, collection_id, "alarm"
        )
        assert len(results) == 2
        nothing = await HybridRetriever(StubEmbedder(axis(0))).retrieve(session, empty.id, "alarm")
        assert nothing == []


async def test_reranker_reorders_and_cuts_off_weak_candidates(session_factory: Factory) -> None:
    async with session_factory() as session:
        collection_id, _ = await seed(session, "manual", CORPUS)
        reranker = StubReranker({"Battery": 0.95, "upper alarm": 0.6, "SMS": 0.2, "Calib": 0.01})
        retriever = HybridRetriever(
            StubEmbedder(axis(0)), reranker=reranker, top_k=3, rerank_min_score=0.1
        )

        results = await retriever.retrieve(session, collection_id, "alarm limit")

        assert [r.content.split()[0] for r in results] == ["Battery", "The", "Alarm"]
        assert [r.rerank_score for r in results] == [0.95, 0.6, 0.2]


async def test_conversation_turns_and_citations_are_stored(session_factory: Factory) -> None:
    async with session_factory() as session, session.begin():
        collection_id, user_id = await seed(session, "manual", CORPUS)
        retriever = HybridRetriever(StubEmbedder(axis(0)), top_k=2)
        chunks = await retriever.retrieve(session, collection_id, "alarm limit")
        answer = Answer(
            text="The limit is 8 degrees. [1]",
            question="alarm limit",
            citations=[Citation(1, chunks[0], supported=True), Citation(2, chunks[1])],
            verified=True,
            model="m",
            input_tokens=100,
            output_tokens=20,
            latency_ms=1234,
        )
        first = await record_turn(
            session,
            user_id=user_id,
            collection_id=collection_id,
            question="What is the alarm limit?",
            answer=answer,
        )
        conversation_id = first.conversation_id

    async with session_factory() as session, session.begin():
        abstention = Answer(text="Not found.", question="q2", abstained=True)
        await record_turn(
            session,
            user_id=user_id,
            collection_id=collection_id,
            question="And the lower limit?",
            answer=abstention,
            conversation_id=conversation_id,
        )

    async with session_factory() as session:
        history = await load_history(session, conversation_id, turns=5)
        assert history == [
            ("user", "What is the alarm limit?"),
            ("assistant", "The limit is 8 degrees. [1]"),
            ("user", "And the lower limit?"),
            ("assistant", "Not found."),
        ]
        assert await load_history(session, conversation_id, turns=1) == history[2:]
        assert await load_history(session, conversation_id, turns=0) == []

        conversation = await session.get_one(models.Conversation, conversation_id)
        assert conversation.title == "What is the alarm limit?"
        stored = (
            await session.scalars(select(models.Citation).order_by(models.Citation.marker))
        ).all()
        assert [(c.marker, c.chunk_id, c.support_score) for c in stored] == [
            (1, chunks[0].chunk_id, 1.0),
            (2, chunks[1].chunk_id, None),
        ]
        message = await session.get_one(models.Message, first.id)
        assert (message.model, message.input_tokens, message.latency_ms) == ("m", 100, 1234)
        abstained = await session.scalar(
            select(func.count()).select_from(models.Message).where(models.Message.abstained)
        )
        assert abstained == 1
