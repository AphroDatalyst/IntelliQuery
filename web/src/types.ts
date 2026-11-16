// Shapes returned by the API (see src/intelliquery/api/schemas.py).

export interface Info {
  version: string;
  chat_model: string | null;
  embedding_model: string;
  max_upload_mb: number;
}

export interface Collection {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  document_count: number;
}

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export interface DocumentItem {
  id: string;
  collection_id: string;
  filename: string;
  status: DocumentStatus;
  size_bytes: number;
  page_count: number | null;
  error: string | null;
  created_at: string;
  chunk_count: number;
}

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Region {
  page: number;
  box: Box | null;
  page_width: number | null;
  page_height: number | null;
}

export interface Citation {
  marker: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  modality: "text" | "table" | "figure";
  page_number: number | null;
  section_path: string | null;
  content: string;
  has_image: boolean;
  regions: Region[];
  supported: boolean | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  abstained: boolean;
  verified: boolean;
  model: string | null;
  latency_ms: number | null;
  citations: Citation[];
  rating: -1 | 1 | null;
}

export interface Conversation {
  id: string;
  collection_id: string;
  title: string | null;
  created_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface AskResponse {
  conversation: Conversation;
  question: Message;
  answer: Message;
  dropped_sentences: number;
}
