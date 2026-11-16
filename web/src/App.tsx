import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError } from "./api";
import { Chat } from "./components/Chat";
import { CitationViewer } from "./components/CitationViewer";
import { Sidebar } from "./components/Sidebar";
import type { Citation, Collection, Conversation, DocumentItem, Info, Message } from "./types";

const STORED_COLLECTION = "intelliquery.collection";
const POLL_MS = 3000;

function describe(error: unknown): string {
  return error instanceof ApiError ? error.message : "Something went wrong.";
}

export function App() {
  const [info, setInfo] = useState<Info | null>(null);
  const [collections, setCollections] = useState<Collection[] | null>(null);
  const [collectionId, setCollectionId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [uploading, setUploading] = useState(0);
  const [cited, setCited] = useState<Citation | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Responses that arrive after the user moved on must not overwrite the new view.
  const current = useRef<{ collection: string | null; conversation: string | null }>({
    collection: null,
    conversation: null,
  });
  current.current = { collection: collectionId, conversation: conversationId };

  const fail = useCallback((problem: unknown) => setError(describe(problem)), []);

  const selectCollection = useCallback((id: string | null) => {
    setCollectionId(id);
    setDocuments([]);
    setConversations([]);
    setConversationId(null);
    setMessages([]);
    setCited(null);
    try {
      if (id) localStorage.setItem(STORED_COLLECTION, id);
    } catch {
      // Storage can be unavailable (private mode); the choice just is not remembered.
    }
  }, []);

  useEffect(() => {
    api.info().then(setInfo).catch(fail);
    api
      .collections()
      .then((loaded) => {
        setCollections(loaded);
        let remembered: string | null = null;
        try {
          remembered = localStorage.getItem(STORED_COLLECTION);
        } catch {
          // See above.
        }
        const initial = loaded.find((c) => c.id === remembered) ?? loaded[0];
        selectCollection(initial?.id ?? null);
      })
      .catch((problem: unknown) => {
        setCollections([]);
        fail(problem);
      });
  }, [fail, selectCollection]);

  const refreshDocuments = useCallback(async (id: string) => {
    const loaded = await api.documents(id);
    if (current.current.collection === id) setDocuments(loaded);
  }, []);

  useEffect(() => {
    if (!collectionId) return;
    refreshDocuments(collectionId).catch(fail);
    api
      .conversations(collectionId)
      .then((loaded) => {
        if (current.current.collection === collectionId) setConversations(loaded);
      })
      .catch(fail);
  }, [collectionId, refreshDocuments, fail]);

  // Ingestion runs in a separate worker; keep asking until every document has settled.
  const ingesting = documents.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!collectionId || !ingesting) return;
    const timer = setInterval(() => {
      refreshDocuments(collectionId).catch(() => undefined);
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [collectionId, ingesting, refreshDocuments]);

  async function createCollection(name: string) {
    try {
      const created = await api.createCollection(name);
      setCollections((existing) => [...(existing ?? []), created]);
      selectCollection(created.id);
    } catch (problem) {
      fail(problem);
    }
  }

  async function deleteCollection(collection: Collection) {
    try {
      await api.deleteCollection(collection.id);
      const remaining = (collections ?? []).filter((c) => c.id !== collection.id);
      setCollections(remaining);
      if (collectionId === collection.id) selectCollection(remaining[0]?.id ?? null);
    } catch (problem) {
      fail(problem);
    }
  }

  async function upload(files: File[]) {
    if (!collectionId || files.length === 0) return;
    const target = collectionId;
    setUploading((n) => n + files.length);
    for (const file of files) {
      try {
        await api.uploadDocument(target, file);
      } catch (problem) {
        setError(`${file.name}: ${describe(problem)}`);
      } finally {
        setUploading((n) => n - 1);
      }
    }
    refreshDocuments(target).catch(fail);
  }

  async function deleteDocument(document: DocumentItem) {
    try {
      await api.deleteDocument(document.id);
      setDocuments((existing) => existing.filter((d) => d.id !== document.id));
      if (cited?.document_id === document.id) setCited(null);
      // Citations of the deleted document are gone from stored answers too.
      if (conversationId) await openConversation(conversationId);
    } catch (problem) {
      fail(problem);
    }
  }

  async function openConversation(id: string) {
    setConversationId(id);
    setCited(null);
    current.current.conversation = id;
    try {
      const detail = await api.conversation(id);
      if (current.current.conversation === id) setMessages(detail.messages);
    } catch (problem) {
      fail(problem);
    }
  }

  function newConversation() {
    setConversationId(null);
    setMessages([]);
    setCited(null);
  }

  async function deleteConversation(conversation: Conversation) {
    try {
      await api.deleteConversation(conversation.id);
      setConversations((existing) => existing.filter((c) => c.id !== conversation.id));
      if (conversationId === conversation.id) newConversation();
    } catch (problem) {
      fail(problem);
    }
  }

  /** Returns false when the question could not be answered, so the caller can keep it. */
  async function ask(question: string): Promise<boolean> {
    if (!collectionId || pendingQuestion !== null) return false;
    const asked = { collection: collectionId, conversation: conversationId };
    setPendingQuestion(question);
    setError(null);
    try {
      const turn = await api.ask(asked.collection, question, asked.conversation);
      if (asked.conversation === null && current.current.collection === asked.collection) {
        setConversations((existing) => [turn.conversation, ...existing]);
      }
      // Show the answer only if the user is still looking at the conversation it belongs to.
      const here = current.current;
      if (here.collection === asked.collection && here.conversation === asked.conversation) {
        setConversationId(turn.conversation.id);
        setMessages((existing) => [...existing, turn.question, turn.answer]);
      }
      return true;
    } catch (problem) {
      fail(problem);
      return false;
    } finally {
      setPendingQuestion(null);
    }
  }

  async function rate(message: Message, rating: -1 | 1) {
    const next = message.rating === rating ? null : rating;
    const apply = (value: Message["rating"]) =>
      setMessages((existing) =>
        existing.map((m) => (m.id === message.id ? { ...m, rating: value } : m)),
      );
    apply(next);
    try {
      await (next === null ? api.clearRating(message.id) : api.rate(message.id, next));
    } catch (problem) {
      apply(message.rating);
      fail(problem);
    }
  }

  const collection = collections?.find((c) => c.id === collectionId) ?? null;

  return (
    <div className={cited ? "layout with-viewer" : "layout"}>
      <Sidebar
        collections={collections}
        collection={collection}
        documents={documents}
        conversations={conversations}
        conversationId={conversationId}
        uploading={uploading}
        maxUploadMb={info?.max_upload_mb ?? null}
        onSelectCollection={selectCollection}
        onCreateCollection={createCollection}
        onDeleteCollection={deleteCollection}
        onUpload={upload}
        onDeleteDocument={deleteDocument}
        onOpenConversation={openConversation}
        onNewConversation={newConversation}
        onDeleteConversation={deleteConversation}
      />
      <Chat
        info={info}
        collection={collection}
        hasCollections={collections === null ? null : collections.length > 0}
        readyDocuments={documents.filter((d) => d.status === "ready").length}
        ingesting={ingesting}
        messages={messages}
        pendingQuestion={pendingQuestion}
        activeCitation={cited}
        error={error}
        onDismissError={() => setError(null)}
        onAsk={ask}
        onCite={setCited}
        onRate={rate}
      />
      {cited && <CitationViewer citation={cited} onClose={() => setCited(null)} />}
    </div>
  );
}
