"""Answer a question from a collection, with every sentence tied to its evidence."""

import json
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.answering import prompts
from intelliquery.answering.citations import Sentence, parse_sentences, render
from intelliquery.config import Settings
from intelliquery.db.models import ChunkModality
from intelliquery.providers import ChatMessage, ChatModel, ChatResult, ProviderError
from intelliquery.providers.openai_compat import (
    HttpReranker,
    OpenAICompatChat,
    OpenAICompatClient,
    OpenAICompatEmbedder,
)
from intelliquery.retrieval import RetrievedChunk, Retriever
from intelliquery.retrieval.search import HybridRetriever
from intelliquery.storage import LocalStorage, ObjectStorage

logger = logging.getLogger(__name__)

History = Sequence[tuple[str, str]]
"""Earlier turns as (role, content), oldest first."""


@dataclass
class Citation:
    marker: int
    chunk: RetrievedChunk
    # True once the verification pass confirmed the chunk supports what cites it;
    # None when verification was disabled or could not be completed.
    supported: bool | None = None


@dataclass
class Answer:
    text: str
    question: str
    """The standalone question that was actually searched for."""
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    verified: bool = False
    # Sentences removed for having no valid citation or failing verification.
    dropped_sentences: int = 0
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class _Usage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, result: ChatResult) -> ChatResult:
        self.input_tokens += result.input_tokens or 0
        self.output_tokens += result.output_tokens or 0
        return result


def parse_verdicts(text: str, claim_count: int) -> list[bool] | None:
    """Read the verifier's JSON. Returns None when it cannot be parsed at all.

    A claim the verifier did not mention is treated as unsupported.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        rows = json.loads(text[start : end + 1])["verdicts"]
        verdicts = [False] * claim_count
        for row in rows:
            index = int(row["claim"]) - 1
            if 0 <= index < claim_count:
                verdicts[index] = row["supported"] is True
    except (ValueError, KeyError, TypeError):
        return None
    return verdicts


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        chat: ChatModel,
        *,
        storage: ObjectStorage | None = None,
        max_tokens: int = 900,
        verify: bool = True,
        send_images: bool = False,
    ) -> None:
        self._retriever = retriever
        self._chat = chat
        self._storage = storage
        self._max_tokens = max_tokens
        self._verify = verify
        self._send_images = send_images

    async def _standalone_question(self, question: str, history: History, usage: _Usage) -> str:
        if not history:
            return question
        result = usage.add(
            await self._chat.complete(
                [
                    ChatMessage("system", prompts.REWRITE_SYSTEM),
                    ChatMessage("user", prompts.rewrite_prompt(history, question)),
                ],
                max_tokens=200,
            )
        )
        # A blank or runaway rewrite is worse than searching for the original words.
        rewritten = result.text.strip().strip('"')
        return rewritten if 0 < len(rewritten) <= 4 * len(question) + 200 else question

    async def _images(self, chunks: Sequence[RetrievedChunk]) -> tuple[list[bytes], str]:
        if not (self._send_images and self._storage):
            return [], ""
        images: list[bytes] = []
        notes: list[str] = []
        for number, chunk in enumerate(chunks, start=1):
            if chunk.asset_key and chunk.modality is not ChunkModality.TEXT:
                try:
                    images.append(await self._storage.get(chunk.asset_key))
                except OSError:
                    logger.warning("missing asset", extra={"asset_key": chunk.asset_key})
                    continue
                notes.append(f"Attached image {len(images)} is source [{number}].")
        return images, "\n".join(notes)

    async def _verify_sentences(
        self, sentences: list[Sentence], chunks: Sequence[RetrievedChunk], usage: _Usage
    ) -> list[bool] | None:
        claims = [(s.text, [chunks[i - 1] for i in s.sources]) for s in sentences]
        messages = [
            ChatMessage("system", prompts.VERIFY_SYSTEM),
            ChatMessage("user", prompts.verify_prompt(claims)),
        ]
        for _attempt in range(2):
            try:
                result = usage.add(
                    await self._chat.complete(
                        messages, max_tokens=60 + 25 * len(claims), json_output=True
                    )
                )
            except ProviderError:
                logger.exception("verification call failed")
                return None
            verdicts = parse_verdicts(result.text, len(claims))
            if verdicts is not None:
                return verdicts
        logger.warning("verifier returned unparseable output")
        return None

    async def answer(
        self,
        session: AsyncSession,
        collection_id: uuid.UUID,
        question: str,
        history: History = (),
    ) -> Answer:
        started = time.perf_counter()
        usage = _Usage()

        def finish(answer: Answer) -> Answer:
            answer.model = self._chat.model
            answer.input_tokens = usage.input_tokens
            answer.output_tokens = usage.output_tokens
            answer.latency_ms = round((time.perf_counter() - started) * 1000)
            logger.info(
                "answered",
                extra={
                    "abstained": answer.abstained,
                    "citations": len(answer.citations),
                    "dropped_sentences": answer.dropped_sentences,
                    "verified": answer.verified,
                    "latency_ms": answer.latency_ms,
                },
            )
            return answer

        def abstain(search_question: str, dropped: int = 0) -> Answer:
            return finish(
                Answer(
                    text=prompts.ABSTAIN_MESSAGE,
                    question=search_question,
                    abstained=True,
                    dropped_sentences=dropped,
                )
            )

        search_question = await self._standalone_question(question, history, usage)
        chunks = await self._retriever.retrieve(session, collection_id, search_question)
        if not chunks:
            # Nothing to ground an answer in: do not ask the model at all.
            return abstain(search_question)

        images, image_notes = await self._images(chunks)
        prompt = prompts.answer_prompt(search_question, chunks)
        if image_notes:
            prompt += "\n\n" + image_notes
        draft = usage.add(
            await self._chat.complete(
                [
                    ChatMessage("system", prompts.ANSWER_SYSTEM),
                    ChatMessage("user", prompt, images=images),
                ],
                max_tokens=self._max_tokens,
            )
        )
        if not draft.text or draft.text.strip(" .`\"'").upper() == prompts.NOT_FOUND:
            return abstain(search_question)

        parsed = parse_sentences(draft.text, source_count=len(chunks))
        # The refusal token mixed into other text is not a sentence of the answer.
        parsed = [s for s in parsed if prompts.NOT_FOUND not in s.text.upper()]
        cited = [s for s in parsed if s.sources]
        dropped = len(parsed) - len(cited)
        if not cited:
            return abstain(search_question, dropped)

        verified = False
        if self._verify:
            verdicts = await self._verify_sentences(cited, chunks, usage)
            if verdicts is not None:
                verified = True
                kept = [s for s, ok in zip(cited, verdicts, strict=True) if ok]
                dropped += len(cited) - len(kept)
                cited = kept
                if not cited:
                    return abstain(search_question, dropped)

        text, order = render(cited)
        citations = [
            Citation(marker=marker, chunk=chunks[source - 1], supported=True if verified else None)
            for marker, source in enumerate(order, start=1)
        ]
        return finish(
            Answer(
                text=text,
                question=search_question,
                citations=citations,
                verified=verified,
                dropped_sentences=dropped,
            )
        )


class ChatModelNotConfigured(RuntimeError):
    pass


def build_answer_service(settings: Settings) -> tuple[AnswerService, list[OpenAICompatClient]]:
    """Wire the answer service from settings. Returns it with the clients to close."""
    if not settings.chat_model:
        raise ChatModelNotConfigured(
            "No chat model is configured. Set INTELLIQUERY_CHAT_MODEL (and "
            "INTELLIQUERY_CHAT_BASE_URL / INTELLIQUERY_CHAT_API_KEY for a hosted provider)."
        )

    def client(base_url: str, key: SecretStr | None) -> OpenAICompatClient:
        return OpenAICompatClient(
            base_url,
            key.get_secret_value() if key else None,
            timeout=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
        )

    embedding_client = client(settings.embedding_base_url, settings.embedding_api_key)
    chat_client = client(settings.chat_base_url, settings.chat_api_key)
    clients = [embedding_client, chat_client]
    reranker = None
    if settings.rerank_base_url and settings.rerank_model:
        rerank_client = client(settings.rerank_base_url, settings.rerank_api_key)
        clients.append(rerank_client)
        reranker = HttpReranker(rerank_client, settings.rerank_model)

    retriever = HybridRetriever(
        OpenAICompatEmbedder(
            embedding_client,
            settings.embedding_model,
            request_dimensions=settings.embedding_request_dimensions,
        ),
        reranker=reranker,
        candidates=settings.retrieval_candidates,
        top_k=settings.retrieval_top_k,
        rerank_min_score=settings.rerank_min_score,
    )
    service = AnswerService(
        retriever,
        OpenAICompatChat(chat_client, settings.chat_model),
        storage=LocalStorage(settings.storage_dir),
        max_tokens=settings.answer_max_tokens,
        verify=settings.answer_verify,
        send_images=settings.chat_send_images,
    )
    return service, clients
