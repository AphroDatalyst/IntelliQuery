import { useRef, useState, type DragEvent, type FormEvent } from "react";

import { formatBytes } from "../citations";
import type { Collection, Conversation, DocumentItem } from "../types";

interface Props {
  collections: Collection[] | null;
  collection: Collection | null;
  documents: DocumentItem[];
  conversations: Conversation[];
  conversationId: string | null;
  uploading: number;
  maxUploadMb: number | null;
  onSelectCollection: (id: string) => void;
  onCreateCollection: (name: string) => Promise<void>;
  onDeleteCollection: (collection: Collection) => void;
  onUpload: (files: File[]) => void;
  onDeleteDocument: (document: DocumentItem) => void;
  onOpenConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (conversation: Conversation) => void;
}

const STATUS_LABEL: Record<DocumentItem["status"], string> = {
  pending: "Queued",
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
};

function documentDetail(document: DocumentItem): string {
  if (document.status === "failed") return document.error ?? "Ingestion failed.";
  if (document.status !== "ready") return `${STATUS_LABEL[document.status]}…`;
  const pages = document.page_count === 1 ? "1 page" : `${document.page_count ?? "?"} pages`;
  return `${pages} · ${formatBytes(document.size_bytes)}`;
}

export function Sidebar(props: Props) {
  const { collections, collection, documents, conversations } = props;
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  async function submitCollection(event: FormEvent) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    await props.onCreateCollection(trimmed);
    setName("");
    setCreating(false);
  }

  function drop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    if (collection) props.onUpload(Array.from(event.dataTransfer.files));
  }

  function confirmDelete(what: string, action: () => void) {
    if (window.confirm(`Delete ${what}? This cannot be undone.`)) action();
  }

  const showForm = creating || collections?.length === 0;

  return (
    <aside className="sidebar">
      <header className="brand">
        <img src="/favicon.svg" alt="" width={22} height={22} />
        <span>IntelliQuery</span>
      </header>

      <section>
        <div className="section-title">
          <h2>Collection</h2>
          {!showForm && (
            <button className="link" onClick={() => setCreating(true)}>
              New
            </button>
          )}
        </div>
        {collections && collections.length > 0 && (
          <div className="row">
            <select
              aria-label="Collection"
              value={collection?.id ?? ""}
              onChange={(event) => props.onSelectCollection(event.target.value)}
            >
              {collections.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
            {collection && (
              <button
                className="icon danger"
                title="Delete collection"
                aria-label={`Delete collection ${collection.name}`}
                onClick={() =>
                  confirmDelete(
                    `the collection "${collection.name}" with all its documents and conversations`,
                    () => props.onDeleteCollection(collection),
                  )
                }
              >
                ×
              </button>
            )}
          </div>
        )}
        {showForm && (
          <form className="row" onSubmit={submitCollection}>
            <input
              autoFocus
              aria-label="New collection name"
              placeholder="Collection name"
              maxLength={200}
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
            <button type="submit" disabled={!name.trim()}>
              Create
            </button>
            {creating && (
              <button type="button" className="link" onClick={() => setCreating(false)}>
                Cancel
              </button>
            )}
          </form>
        )}
      </section>

      {collection && (
        <>
          <section
            className={dragging ? "documents dragging" : "documents"}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={drop}
          >
            <div className="section-title">
              <h2>Documents</h2>
              <button className="link" onClick={() => fileInput.current?.click()}>
                Upload PDF
              </button>
              <input
                ref={fileInput}
                type="file"
                accept="application/pdf,.pdf"
                multiple
                hidden
                onChange={(event) => {
                  props.onUpload(Array.from(event.target.files ?? []));
                  event.target.value = "";
                }}
              />
            </div>
            {documents.length === 0 && props.uploading === 0 && (
              <p className="hint">
                Drop PDF files here or use Upload PDF
                {props.maxUploadMb ? ` (up to ${props.maxUploadMb} MB each)` : ""}.
              </p>
            )}
            <ul className="list">
              {documents.map((document) => (
                <li key={document.id} className="item">
                  <span
                    className={`dot ${document.status}`}
                    title={STATUS_LABEL[document.status]}
                  />
                  <div className="item-text">
                    <span className="item-title" title={document.filename}>
                      {document.filename}
                    </span>
                    <span className={document.status === "failed" ? "item-sub failed" : "item-sub"}>
                      {documentDetail(document)}
                    </span>
                  </div>
                  <button
                    className="icon danger"
                    title="Delete document"
                    aria-label={`Delete ${document.filename}`}
                    onClick={() =>
                      confirmDelete(`"${document.filename}"`, () =>
                        props.onDeleteDocument(document),
                      )
                    }
                  >
                    ×
                  </button>
                </li>
              ))}
              {props.uploading > 0 && (
                <li className="item">
                  <span className="dot processing" />
                  <span className="item-sub">
                    Uploading {props.uploading} file{props.uploading === 1 ? "" : "s"}…
                  </span>
                </li>
              )}
            </ul>
          </section>

          <section className="conversations">
            <div className="section-title">
              <h2>Conversations</h2>
              <button className="link" onClick={props.onNewConversation}>
                New chat
              </button>
            </div>
            {conversations.length === 0 && <p className="hint">No conversations yet.</p>}
            <ul className="list">
              {conversations.map((conversation) => (
                <li
                  key={conversation.id}
                  className={conversation.id === props.conversationId ? "item active" : "item"}
                >
                  <button
                    className="item-button"
                    title={conversation.title ?? undefined}
                    onClick={() => props.onOpenConversation(conversation.id)}
                  >
                    {conversation.title ?? "Untitled"}
                  </button>
                  <button
                    className="icon danger"
                    title="Delete conversation"
                    aria-label="Delete conversation"
                    onClick={() =>
                      confirmDelete("this conversation", () =>
                        props.onDeleteConversation(conversation),
                      )
                    }
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </aside>
  );
}
