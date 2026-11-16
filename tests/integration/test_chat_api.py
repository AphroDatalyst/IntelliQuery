"""The HTTP API end to end, against a real database with the models scripted."""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelliquery.answering import prompts
from intelliquery.answering.service import AnswerService
from intelliquery.api.app import create_app
from intelliquery.api.deps import get_answer_service
from intelliquery.config import Settings
from intelliquery.db import models
from intelliquery.db.base import Base
from intelliquery.db.models import ChunkModality, DocumentStatus
from intelliquery.ingestion.service import asset_key, page_key, source_key
from intelliquery.providers import ProviderError
from intelliquery.retrieval.search import HybridRetriever
from intelliquery.storage import LocalStorage
from tests.fakes import PNG
from tests.integration.test_retrieval import StubEmbedder, axis
from tests.unit.test_answering import ScriptedChat, verdicts

pytestmark = pytest.mark.integration

PDF = b"%PDF-1.7 a manual"
MISSING = uuid.uuid4()


@dataclass
class Api:
    client: httpx.AsyncClient
    app: FastAPI
    factory: async_sessionmaker[AsyncSession]
    storage: LocalStorage

    def script(self, *replies: str | Exception) -> ScriptedChat:
        """Answer the next questions with these model replies."""
        chat = ScriptedChat(*replies)
        service = AnswerService(HybridRetriever(StubEmbedder(axis(0)), top_k=2), chat)
        self.app.dependency_overrides[get_answer_service] = lambda: service
        return chat

    async def collection(self, name: str = "manuals") -> str:
        response = await self.client.post("/api/collections", json={"name": name})
        assert response.status_code == 201
        return str(response.json()["id"])

    async def upload(self, collection_id: str, data: bytes = PDF, name: str = "manual.pdf"):  # type: ignore[no-untyped-def]
        return await self.client.post(
            f"/api/collections/{collection_id}/documents",
            files={"file": (name, data, "application/pdf")},
        )

    async def indexed_document(self, collection_id: str) -> str:
        """Upload a document and store what ingestion would have produced for it."""
        document_id = uuid.UUID((await self.upload(collection_id)).json()["id"])
        await self.storage.put(page_key(document_id, 4), PNG)
        await self.storage.put(asset_key(document_id, 1), PNG)
        async with self.factory() as session, session.begin():
            document = await session.get_one(models.Document, document_id)
            document.status = DocumentStatus.READY
            document.page_count = 9
            session.add(
                models.DocumentPage(
                    document_id=document_id,
                    page_number=4,
                    image_key=page_key(document_id, 4),
                    width=800,
                    height=1000,
                )
            )
            box = {"x0": 10.0, "y0": 20.0, "x1": 200.0, "y1": 60.0}
            session.add_all(
                [
                    models.Chunk(
                        document_id=document_id,
                        collection_id=uuid.UUID(collection_id),
                        chunk_index=0,
                        modality=ChunkModality.TEXT,
                        page_number=4,
                        bbox=box,
                        section_path="Alarms > Limits",
                        content="The upper alarm limit for the cold room is 8 degrees.",
                        embedding=axis(0),
                        meta={"regions": [{"page": 4, "bbox": box}]},
                    ),
                    models.Chunk(
                        document_id=document_id,
                        collection_id=uuid.UUID(collection_id),
                        chunk_index=1,
                        modality=ChunkModality.TABLE,
                        page_number=9,
                        content="| Channel | Alarm |\n|---|---|\n| SMS | yes |",
                        asset_key=asset_key(document_id, 1),
                        embedding=axis(1),
                    ),
                ]
            )
        return str(document_id)

    async def count(self, model: type[Base]) -> int:
        async with self.factory() as session:
            return await session.scalar(select(func.count()).select_from(model)) or 0


@pytest.fixture
async def api(
    session_factory: async_sessionmaker[AsyncSession], storage: LocalStorage
) -> AsyncIterator[Api]:
    app = create_app(Settings(environment="test", ingest_max_upload_mb=1))
    # The lifespan is not run under the test transport; provide what it would.
    app.state.session_factory = session_factory
    app.state.storage = storage
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
        yield Api(client, app, session_factory, storage)


async def test_collections_can_be_created_listed_and_deleted(api: Api) -> None:
    collection_id = await api.collection("manuals")
    await api.upload(collection_id)

    listed = (await api.client.get("/api/collections")).json()
    assert [(c["name"], c["document_count"]) for c in listed] == [("manuals", 1)]

    duplicate = await api.client.post("/api/collections", json={"name": " manuals "})
    assert duplicate.status_code == 409
    assert (await api.client.post("/api/collections", json={"name": "  "})).status_code == 422

    assert (await api.client.delete(f"/api/collections/{collection_id}")).status_code == 204
    assert (await api.client.get("/api/collections")).json() == []
    assert await api.count(models.Document) == 0
    assert (await api.client.delete(f"/api/collections/{collection_id}")).status_code == 404


async def test_upload_queues_a_document_and_is_idempotent(api: Api) -> None:
    collection_id = await api.collection()

    first = await api.upload(collection_id, name="C:\\docs\\manual.pdf")
    assert first.status_code == 201
    document = first.json()
    assert (document["filename"], document["status"], document["chunk_count"]) == (
        "manual.pdf",
        "pending",
        0,
    )
    assert await api.storage.get(source_key(uuid.UUID(document["id"]))) == PDF
    assert await api.count(models.IngestionJob) == 1

    again = await api.upload(collection_id, name="renamed.pdf")
    assert again.status_code == 200
    assert again.json()["id"] == document["id"]
    assert await api.count(models.IngestionJob) == 1

    listed = (await api.client.get(f"/api/collections/{collection_id}/documents")).json()
    assert [d["id"] for d in listed] == [document["id"]]
    assert (await api.client.get(f"/api/documents/{document['id']}")).json() == listed[0]


async def test_upload_rejects_what_cannot_be_ingested(api: Api) -> None:
    collection_id = await api.collection()

    not_a_pdf = await api.upload(collection_id, data=b"MZ executable", name="evil.pdf")
    assert not_a_pdf.status_code == 415
    assert not_a_pdf.json() == {"detail": "Only PDF files are supported."}

    too_large = await api.upload(collection_id, data=PDF + b"0" * (1024 * 1024))
    assert too_large.status_code == 413

    assert (await api.upload(str(MISSING))).status_code == 404
    assert await api.count(models.Document) == 0


async def test_deleting_a_document_removes_its_rows_and_files(api: Api) -> None:
    collection_id = await api.collection()
    document_id = await api.indexed_document(collection_id)

    assert (await api.client.delete(f"/api/documents/{document_id}")).status_code == 204

    assert await api.count(models.Chunk) == 0
    assert await api.count(models.DocumentPage) == 0
    with pytest.raises(FileNotFoundError):
        await api.storage.get(source_key(uuid.UUID(document_id)))
    assert (await api.client.get(f"/api/documents/{document_id}")).status_code == 404


async def test_answer_comes_back_with_citations_that_locate_the_evidence(api: Api) -> None:
    collection_id = await api.collection()
    document_id = await api.indexed_document(collection_id)
    api.script("The limit is 8 degrees [1]. Alarms go out by SMS [2].", verdicts(True, True))

    response = await api.client.post(
        f"/api/collections/{collection_id}/ask", json={"question": " What is the alarm limit? "}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"]["content"] == "What is the alarm limit?"
    answer = body["answer"]
    assert answer["content"] == "The limit is 8 degrees. [1] Alarms go out by SMS. [2]"
    assert (answer["abstained"], answer["verified"], body["dropped_sentences"]) == (False, True, 0)
    text_citation, table_citation = answer["citations"]
    assert text_citation == {
        "marker": 1,
        "chunk_id": text_citation["chunk_id"],
        "document_id": document_id,
        "filename": "manual.pdf",
        "modality": "text",
        "page_number": 4,
        "section_path": "Alarms > Limits",
        "content": "The upper alarm limit for the cold room is 8 degrees.",
        "has_image": False,
        "regions": [
            {
                "page": 4,
                "box": {"x0": 10.0, "y0": 20.0, "x1": 200.0, "y1": 60.0},
                "page_width": 800,
                "page_height": 1000,
            }
        ],
        "supported": True,
    }
    # No stored regions and no rendered page: located by page number alone.
    assert table_citation["regions"] == [
        {"page": 9, "box": None, "page_width": None, "page_height": None}
    ]
    assert table_citation["has_image"] is True

    page = await api.client.get(f"/api/documents/{document_id}/pages/4/image")
    assert (page.status_code, page.headers["content-type"], page.content) == (200, "image/png", PNG)
    assert (await api.client.get(f"/api/documents/{document_id}/pages/5/image")).status_code == 404
    crop = await api.client.get(f"/api/chunks/{table_citation['chunk_id']}/image")
    assert crop.content == PNG
    no_crop = await api.client.get(f"/api/chunks/{text_citation['chunk_id']}/image")
    assert no_crop.status_code == 404


async def test_follow_up_continues_the_conversation(api: Api) -> None:
    collection_id = await api.collection()
    await api.indexed_document(collection_id)
    chat = api.script(
        "The limit is 8 degrees [1].",
        verdicts(True),
        "How are alarm notifications delivered?",  # the follow-up, rewritten as standalone
        "They are sent by SMS [2].",
        verdicts(True),
    )
    ask = f"/api/collections/{collection_id}/ask"

    first = (await api.client.post(ask, json={"question": "What is the alarm limit?"})).json()
    conversation_id = first["conversation"]["id"]
    second = await api.client.post(
        ask, json={"question": "How is it delivered?", "conversation_id": conversation_id}
    )

    assert second.json()["conversation"]["id"] == conversation_id
    assert "What is the alarm limit?" in chat.calls[2][1].content
    listed = (await api.client.get(f"/api/collections/{collection_id}/conversations")).json()
    assert [(c["id"], c["title"]) for c in listed] == [
        (conversation_id, "What is the alarm limit?")
    ]
    detail = (await api.client.get(f"/api/conversations/{conversation_id}")).json()
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "What is the alarm limit?"),
        ("assistant", "The limit is 8 degrees. [1]"),
        ("user", "How is it delivered?"),
        ("assistant", "They are sent by SMS. [1]"),
    ]
    assert detail["messages"][3] == second.json()["answer"]

    other = await api.collection("other")
    crossed = await api.client.post(
        f"/api/collections/{other}/ask",
        json={"question": "And now?", "conversation_id": conversation_id},
    )
    assert crossed.status_code == 404

    assert (await api.client.delete(f"/api/conversations/{conversation_id}")).status_code == 204
    assert (await api.client.get(f"/api/conversations/{conversation_id}")).status_code == 404
    assert await api.count(models.Message) == 0


async def test_question_without_evidence_is_declined_and_recorded(api: Api) -> None:
    collection_id = await api.collection()
    chat = api.script()  # an empty collection must not reach the model at all

    response = await api.client.post(
        f"/api/collections/{collection_id}/ask", json={"question": "What is the alarm limit?"}
    )

    answer = response.json()["answer"]
    assert (answer["content"], answer["abstained"], answer["verified"], answer["citations"]) == (
        prompts.ABSTAIN_MESSAGE,
        True,
        False,
        [],
    )
    assert chat.calls == []
    assert await api.count(models.Message) == 2


async def test_model_failures_are_reported_and_store_nothing(api: Api) -> None:
    collection_id = await api.collection()
    await api.indexed_document(collection_id)
    ask = f"/api/collections/{collection_id}/ask"

    api.script(ProviderError("upstream timed out"))
    failed = await api.client.post(ask, json={"question": "What is the alarm limit?"})
    assert failed.status_code == 502
    assert "upstream" not in failed.text
    assert await api.count(models.Conversation) == 0

    # With no chat model configured, the API says what to set instead of failing obscurely.
    api.app.dependency_overrides.clear()
    unconfigured = await api.client.post(ask, json={"question": "What is the alarm limit?"})
    assert unconfigured.status_code == 503
    assert "INTELLIQUERY_CHAT_MODEL" in unconfigured.json()["detail"]
    assert (await api.client.get("/api/info")).json()["chat_model"] is None


async def test_an_answer_can_be_rated_and_the_vote_changed(api: Api) -> None:
    collection_id = await api.collection()
    await api.indexed_document(collection_id)
    api.script("The limit is 8 degrees [1].", verdicts(True))
    turn = (
        await api.client.post(
            f"/api/collections/{collection_id}/ask", json={"question": "What is the limit?"}
        )
    ).json()
    conversation = f"/api/conversations/{turn['conversation']['id']}"
    feedback = f"/api/messages/{turn['answer']['id']}/feedback"

    async def rating() -> int | None:
        messages = (await api.client.get(conversation)).json()["messages"]
        return messages[1]["rating"]  # type: ignore[no-any-return]

    assert await rating() is None
    assert (await api.client.put(feedback, json={"rating": 1})).status_code == 204
    assert await rating() == 1
    changed = await api.client.put(feedback, json={"rating": -1, "comment": "wrong page"})
    assert changed.status_code == 204
    assert await rating() == -1
    assert await api.count(models.Feedback) == 1
    assert (await api.client.put(feedback, json={"rating": 5})).status_code == 422

    # Only answers can be rated.
    question = f"/api/messages/{turn['question']['id']}/feedback"
    assert (await api.client.put(question, json={"rating": 1})).status_code == 404

    assert (await api.client.delete(feedback)).status_code == 204
    assert await rating() is None
