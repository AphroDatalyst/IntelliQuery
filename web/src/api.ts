import type {
  AskResponse,
  Collection,
  Conversation,
  ConversationDetail,
  DocumentItem,
  Info,
} from "./types";

/** An error whose message is safe to show to the user. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    // Validation errors arrive as a list of {msg, loc}.
    if (Array.isArray(detail) && typeof detail[0]?.msg === "string") return detail[0].msg;
  } catch {
    // Not JSON: fall through to the generic message.
  }
  return `Request failed (${response.status}).`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError("Cannot reach the server.", 0);
  }
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

function json(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export const api = {
  info: () => request<Info>("/info"),

  collections: () => request<Collection[]>("/collections"),
  createCollection: (name: string) => request<Collection>("/collections", json("POST", { name })),
  deleteCollection: (id: string) => request<void>(`/collections/${id}`, { method: "DELETE" }),

  documents: (collectionId: string) =>
    request<DocumentItem[]>(`/collections/${collectionId}/documents`),
  uploadDocument: (collectionId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentItem>(`/collections/${collectionId}/documents`, {
      method: "POST",
      body: form,
    });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),

  conversations: (collectionId: string) =>
    request<Conversation[]>(`/collections/${collectionId}/conversations`),
  conversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: "DELETE" }),

  ask: (collectionId: string, question: string, conversationId: string | null) =>
    request<AskResponse>(
      `/collections/${collectionId}/ask`,
      json("POST", { question, conversation_id: conversationId }),
    ),

  rate: (messageId: string, rating: -1 | 1) =>
    request<void>(`/messages/${messageId}/feedback`, json("PUT", { rating })),
  clearRating: (messageId: string) =>
    request<void>(`/messages/${messageId}/feedback`, { method: "DELETE" }),

  pageImageUrl: (documentId: string, page: number) =>
    `/api/documents/${documentId}/pages/${page}/image`,
  chunkImageUrl: (chunkId: string) => `/api/chunks/${chunkId}/image`,
};
