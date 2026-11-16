import { useEffect, useRef, useState, type KeyboardEvent } from "react";

import { splitAnswer } from "../citations";
import type { Citation, Collection, Info, Message } from "../types";

interface Props {
  info: Info | null;
  collection: Collection | null;
  /** Null while the collections are still loading. */
  hasCollections: boolean | null;
  readyDocuments: number;
  ingesting: boolean;
  messages: Message[];
  pendingQuestion: string | null;
  activeCitation: Citation | null;
  error: string | null;
  onDismissError: () => void;
  onAsk: (question: string) => Promise<boolean>;
  onCite: (citation: Citation) => void;
  onRate: (message: Message, rating: -1 | 1) => void;
}

function sourceLabel(citation: Citation): string {
  const page = citation.page_number === null ? "" : `, p. ${citation.page_number}`;
  return `${citation.filename}${page}`;
}

function Answer({ message, activeCitation, onCite, onRate }: AnswerProps) {
  const byMarker = new Map(message.citations.map((c) => [c.marker, c]));
  const markers = new Set(byMarker.keys());
  const isActive = (citation: Citation) => activeCitation?.chunk_id === citation.chunk_id;

  let status: { label: string; tone: string; title: string };
  if (message.abstained) {
    status = {
      label: "Not in the documents",
      tone: "muted",
      title: "The documents did not support an answer, so none was given.",
    };
  } else if (message.verified) {
    status = {
      label: "Verified",
      tone: "good",
      title: "Each sentence was checked against the sources it cites.",
    };
  } else {
    status = {
      label: "Unverified",
      tone: "warn",
      title: "Every sentence cites a source, but the verification pass did not complete.",
    };
  }

  return (
    <div className="message assistant">
      <div className="answer-text">
        {message.content.split("\n").map((line, index) => (
          <p key={index}>
            {splitAnswer(line, markers).map((segment, position) => {
              if (segment.kind === "text") return <span key={position}>{segment.text}</span>;
              const citation = byMarker.get(segment.marker)!;
              return (
                <button
                  key={position}
                  className={isActive(citation) ? "marker active" : "marker"}
                  title={sourceLabel(citation)}
                  aria-label={`Source ${segment.marker}: ${sourceLabel(citation)}`}
                  onClick={() => onCite(citation)}
                >
                  {segment.marker}
                </button>
              );
            })}
          </p>
        ))}
      </div>

      {message.citations.length > 0 && (
        <ul className="sources">
          {message.citations.map((citation) => (
            <li key={citation.marker}>
              <button
                className={isActive(citation) ? "source active" : "source"}
                onClick={() => onCite(citation)}
              >
                <span className="source-marker">{citation.marker}</span>
                <span className="source-name">{sourceLabel(citation)}</span>
                {citation.modality !== "text" && (
                  <span className="source-kind">{citation.modality}</span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}

      <footer className="answer-meta">
        <span className={`badge ${status.tone}`} title={status.title}>
          {status.label}
        </span>
        {message.latency_ms !== null && (
          <span className="muted">{(message.latency_ms / 1000).toFixed(1)} s</span>
        )}
        <span className="spacer" />
        <button
          className={message.rating === 1 ? "vote active" : "vote"}
          aria-label="Good answer"
          aria-pressed={message.rating === 1}
          title="Good answer"
          onClick={() => onRate(message, 1)}
        >
          👍
        </button>
        <button
          className={message.rating === -1 ? "vote active" : "vote"}
          aria-label="Bad answer"
          aria-pressed={message.rating === -1}
          title="Bad answer"
          onClick={() => onRate(message, -1)}
        >
          👎
        </button>
      </footer>
    </div>
  );
}

interface AnswerProps {
  message: Message;
  activeCitation: Citation | null;
  onCite: (citation: Citation) => void;
  onRate: (message: Message, rating: -1 | 1) => void;
}

export function Chat(props: Props) {
  const { info, collection, messages, pendingQuestion } = props;
  const [draft, setDraft] = useState("");
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages.length, pendingQuestion]);

  // Why a question cannot be asked right now, if it cannot.
  let blocked: string | null = null;
  if (props.hasCollections === false) {
    blocked = "Create a collection to get started.";
  } else if (!collection) {
    blocked = "";
  } else if (info && !info.chat_model) {
    blocked =
      "No chat model is configured, so questions cannot be answered yet. " +
      "Set INTELLIQUERY_CHAT_MODEL and restart the server.";
  } else if (props.readyDocuments === 0) {
    blocked = props.ingesting
      ? "Your documents are still being processed. You can ask as soon as one is ready."
      : "Upload a PDF to this collection, then ask a question about it.";
  }
  const canAsk = blocked === null && pendingQuestion === null;

  function send() {
    const question = draft.trim();
    if (!question || !canAsk) return;
    setDraft("");
    // A question that failed goes back into the box rather than being lost.
    void props.onAsk(question).then((answered) => {
      if (!answered) setDraft((current) => current || question);
    });
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send();
    }
  }

  const empty = messages.length === 0 && pendingQuestion === null;

  return (
    <main className="chat">
      <header className="chat-header">
        <h1>{collection?.name ?? "IntelliQuery"}</h1>
        {info?.chat_model && <span className="muted">{info.chat_model}</span>}
      </header>

      <div className="messages" aria-live="polite">
        {empty && (
          <div className="empty">
            <h2>Ask your documents</h2>
            <p>
              Answers are written only from what the documents say. Every sentence links to the
              page region that supports it, and when the documents do not contain the answer you
              are told so.
            </p>
            {blocked && <p className="notice">{blocked}</p>}
          </div>
        )}
        {messages.map((message) =>
          message.role === "user" ? (
            <div key={message.id} className="message user">
              {message.content}
            </div>
          ) : (
            <Answer
              key={message.id}
              message={message}
              activeCitation={props.activeCitation}
              onCite={props.onCite}
              onRate={props.onRate}
            />
          ),
        )}
        {pendingQuestion !== null && (
          <>
            <div className="message user">{pendingQuestion}</div>
            <div className="message assistant thinking">
              <span className="spinner" /> Searching the documents and checking the answer…
            </div>
          </>
        )}
        <div ref={end} />
      </div>

      {props.error && (
        <div className="error" role="alert">
          <span>{props.error}</span>
          <button className="icon" aria-label="Dismiss" onClick={props.onDismissError}>
            ×
          </button>
        </div>
      )}

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          send();
        }}
      >
        <textarea
          aria-label="Question"
          rows={2}
          maxLength={4000}
          placeholder={
            blocked === null ? "Ask a question about these documents" : "Asking is not available yet"
          }
          value={draft}
          disabled={blocked !== null}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <button type="submit" className="primary" disabled={!canAsk || !draft.trim()}>
          Ask
        </button>
      </form>
      {!empty && blocked && <p className="notice below">{blocked}</p>}
    </main>
  );
}
