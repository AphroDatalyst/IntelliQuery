"""The answer service's grounding rules, with retrieval and the model scripted."""

import uuid
from collections.abc import Sequence
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.answering import prompts
from intelliquery.answering.service import AnswerService, parse_verdicts
from intelliquery.db.models import ChunkModality
from intelliquery.providers import ChatMessage, ChatResult, ProviderError
from intelliquery.retrieval import RetrievedChunk
from intelliquery.storage import LocalStorage

SESSION = cast(AsyncSession, None)  # the scripted retriever never touches the database
COLLECTION = uuid.uuid4()


def chunk(content: str, page: int = 1, **kwargs: object) -> RetrievedChunk:
    defaults: dict[str, object] = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "filename": "manual.pdf",
        "modality": ChunkModality.TEXT,
        "content": content,
        "page_number": page,
    }
    return RetrievedChunk(**{**defaults, **kwargs})  # type: ignore[arg-type]


CHUNKS = [
    chunk("The upper alarm limit is 8 C.", 4),
    chunk("Alarms are sent by SMS and e-mail.", 9),
    chunk("The base station stores 40,000 readings.", 12),
]


class ScriptedRetriever:
    def __init__(self, chunks: Sequence[RetrievedChunk] = CHUNKS) -> None:
        self._chunks = list(chunks)
        self.queries: list[str] = []

    async def retrieve(
        self, session: AsyncSession, collection_id: uuid.UUID, query: str
    ) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self._chunks


class ScriptedChat:
    model = "scripted"

    def __init__(self, *replies: str | Exception) -> None:
        self._replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, json_output: bool = False
    ) -> ChatResult:
        self.calls.append(list(messages))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(reply, input_tokens=10, output_tokens=5)


def verdicts(*supported: bool) -> str:
    rows = ", ".join(
        f'{{"claim": {i}, "supported": {str(ok).lower()}}}' for i, ok in enumerate(supported, 1)
    )
    return f'{{"verdicts": [{rows}]}}'


async def ask(chat: ScriptedChat, retriever: ScriptedRetriever | None = None, **kwargs: object):  # type: ignore[no-untyped-def]
    service = AnswerService(retriever or ScriptedRetriever(), chat, **kwargs)  # type: ignore[arg-type]
    return await service.answer(SESSION, COLLECTION, "What is the alarm limit?")


async def test_supported_answer_is_returned_with_renumbered_citations() -> None:
    chat = ScriptedChat("Alarms go out by SMS [2]. The limit is 8 C [1].", verdicts(True, True))
    answer = await ask(chat)

    assert answer.text == "Alarms go out by SMS. [1] The limit is 8 C. [2]"
    assert [(c.marker, c.chunk.page_number, c.supported) for c in answer.citations] == [
        (1, 9, True),
        (2, 4, True),
    ]
    assert (answer.abstained, answer.verified, answer.dropped_sentences) == (False, True, 0)
    assert (answer.input_tokens, answer.output_tokens, answer.model) == (20, 10, "scripted")


async def test_sources_are_numbered_in_the_prompt_and_rules_are_in_the_system_message() -> None:
    chat = ScriptedChat("The limit is 8 C [1].", verdicts(True))
    await ask(chat)

    system, user = chat.calls[0]
    assert "NOT_FOUND" in system.content
    assert "[1] manual.pdf, page 4 (text)\nThe upper alarm limit is 8 C." in user.content
    assert "[3] manual.pdf, page 12 (text)" in user.content
    assert user.content.endswith("Question: What is the alarm limit?")


async def test_no_retrieved_chunks_means_abstaining_without_calling_the_model() -> None:
    chat = ScriptedChat()
    answer = await ask(chat, ScriptedRetriever([]))

    assert answer.abstained
    assert answer.text == prompts.ABSTAIN_MESSAGE
    assert chat.calls == []


@pytest.mark.parametrize("reply", ["NOT_FOUND", "not_found.", "`NOT_FOUND`", ""])
async def test_model_refusal_becomes_an_abstention(reply: str) -> None:
    answer = await ask(ScriptedChat(reply))
    assert answer.abstained
    assert answer.citations == []


async def test_uncited_sentences_are_dropped() -> None:
    chat = ScriptedChat(
        "The limit is 8 C [1]. This is typical for cold storage. Enjoy!", verdicts(True)
    )
    answer = await ask(chat)

    assert answer.text == "The limit is 8 C. [1]"
    assert answer.dropped_sentences == 2


async def test_answer_with_no_valid_citation_at_all_is_an_abstention() -> None:
    answer = await ask(ScriptedChat("The limit is probably 10 C. See the manual [9]."))
    assert answer.abstained
    assert answer.dropped_sentences == 2


async def test_sentences_failing_verification_are_removed() -> None:
    chat = ScriptedChat(
        "The limit is 8 C [1]. The device is waterproof to 30 m [3].", verdicts(True, False)
    )
    answer = await ask(chat)

    assert answer.text == "The limit is 8 C. [1]"
    assert [c.chunk.page_number for c in answer.citations] == [4]
    assert answer.dropped_sentences == 1

    _system, verify_user = chat.calls[1]
    assert "Claim 2: The device is waterproof to 30 m." in verify_user.content
    assert "The base station stores 40,000 readings." in verify_user.content


async def test_abstains_when_verification_rejects_everything() -> None:
    answer = await ask(ScriptedChat("The limit is 12 C [1].", verdicts(False)))
    assert answer.abstained
    assert answer.verified is False
    assert answer.dropped_sentences == 1


async def test_claim_missing_from_the_verdicts_is_treated_as_unsupported() -> None:
    chat = ScriptedChat("The limit is 8 C [1]. SMS is used [2].", verdicts(True))
    answer = await ask(chat)
    assert answer.text == "The limit is 8 C. [1]"


async def test_unparseable_verifier_output_is_retried_then_flagged_unverified() -> None:
    chat = ScriptedChat("The limit is 8 C [1].", "I think it is fine", "still not json")
    answer = await ask(chat)

    assert len(chat.calls) == 3
    assert answer.text == "The limit is 8 C. [1]"
    assert answer.verified is False
    assert answer.citations[0].supported is None


async def test_verifier_failure_does_not_lose_the_answer_but_marks_it_unverified() -> None:
    answer = await ask(ScriptedChat("The limit is 8 C [1].", ProviderError("down")))
    assert (answer.abstained, answer.verified) == (False, False)


async def test_verification_can_be_disabled() -> None:
    chat = ScriptedChat("The limit is 8 C [1].")
    answer = await ask(chat, verify=False)
    assert len(chat.calls) == 1
    assert answer.verified is False


async def test_follow_up_is_rewritten_into_a_standalone_question() -> None:
    chat = ScriptedChat(
        "How are alarms for the upper limit delivered?", "By SMS [2].", verdicts(True)
    )
    retriever = ScriptedRetriever()
    service = AnswerService(retriever, chat)
    history = [("user", "What is the alarm limit?"), ("assistant", "8 C. [1]")]

    answer = await service.answer(SESSION, COLLECTION, "How is it delivered?", history)

    assert retriever.queries == ["How are alarms for the upper limit delivered?"]
    assert answer.question == "How are alarms for the upper limit delivered?"
    rewrite_user = chat.calls[0][1].content
    assert "assistant: 8 C. [1]" in rewrite_user
    assert rewrite_user.endswith("Latest message: How is it delivered?")
    assert "Question: How are alarms for the upper limit delivered?" in chat.calls[1][1].content


async def test_blank_rewrite_falls_back_to_the_original_question() -> None:
    chat = ScriptedChat("   ", "By SMS [2].", verdicts(True))
    retriever = ScriptedRetriever()
    await AnswerService(retriever, chat).answer(
        SESSION, COLLECTION, "How is it delivered?", [("user", "x"), ("assistant", "y")]
    )
    assert retriever.queries == ["How is it delivered?"]


async def test_figure_and_table_images_are_attached_when_enabled(storage: LocalStorage) -> None:
    await storage.put("assets/fig.png", b"figure-bytes")
    chunks = [
        chunk("Text only."),
        chunk("Wiring diagram.", modality=ChunkModality.FIGURE, asset_key="assets/fig.png"),
        chunk("Missing crop.", modality=ChunkModality.TABLE, asset_key="assets/gone.png"),
    ]
    chat = ScriptedChat("See the wiring [2].", verdicts(True))
    await ask(chat, ScriptedRetriever(chunks), storage=storage, send_images=True)

    user = chat.calls[0][1]
    assert user.images == [b"figure-bytes"]
    assert "Attached image 1 is source [2]." in user.content


async def test_images_are_not_sent_by_default(storage: LocalStorage) -> None:
    await storage.put("assets/fig.png", b"figure-bytes")
    chunks = [chunk("Diagram.", modality=ChunkModality.FIGURE, asset_key="assets/fig.png")]
    chat = ScriptedChat("See it [1].", verdicts(True))
    await ask(chat, ScriptedRetriever(chunks), storage=storage)
    assert chat.calls[0][1].images == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"verdicts": [{"claim": 1, "supported": true}]}', [True, False]),
        ('```json\n{"verdicts": [{"claim": 2, "supported": true}]}\n```', [False, True]),
        ('{"verdicts": [{"claim": 1, "supported": "yes"}]}', [False, False]),
        ('{"verdicts": [{"claim": 5, "supported": true}]}', [False, False]),
        ("no json here", None),
        ('{"something": "else"}', None),
        ("{broken", None),
    ],
)
def test_parse_verdicts(text: str, expected: list[bool] | None) -> None:
    assert parse_verdicts(text, 2) == expected
