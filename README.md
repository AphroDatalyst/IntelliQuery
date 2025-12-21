# IntelliQuery - Local RAG (Ollama)
Local **RAG system** over PDFs with conversation memory, hybrid search, re-ranking and citation-grounded answers.

* Pipeline: PDF ingestion → chunking → embedding (Ollama) → Chroma vector store → retrieval → rerank → LLM generation → strict fallback

## Features

* PDF ingestion with chunking
* Dense + hybrid retrieval (BM25 + embeddings)
* Multi-turn memory
* Query normalization & expansion
* Sentence-level citations
* Retrieval confidence & grounding validation
* Streamlit chat UI with persistent history

**Metrics:** Precision@k, Recall@k, MRR, Answer Relevance & Groundedness

## Quick Start

```bash
pip install -r requirements.txt
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
streamlit run app.py
```
* Upload PDFs → Build Index → Ask Questions


## Next branch (planned)
* multi-modal-rag → add image/PDF-table ingestion