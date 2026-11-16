# IntelliQuery

[![CI](https://github.com/AphroDatalyst/IntelliQuery/actions/workflows/ci.yml/badge.svg)](https://github.com/AphroDatalyst/IntelliQuery/actions/workflows/ci.yml)

Ask questions about your documents and get answers you can check. IntelliQuery is a
multimodal retrieval-augmented generation (RAG) service: it reads text, tables and
figures from uploaded files, answers only from what it retrieved, and attaches every
claim to the page region that supports it. When the documents do not contain the
answer, it says so.

## Status

The project is being rebuilt from a local Streamlit prototype into a production
service. This table is kept current with what is actually in the repository.

| Area | State |
|---|---|
| Service foundation: API skeleton, config, structured logging, health probes | Done |
| PostgreSQL + pgvector schema, migrations, CI, container image | Done |
| Multimodal ingestion: layout-aware parsing, tables, figure descriptions, job queue, worker | Done |
| Hybrid retrieval, reranking, cited and verified answers, abstention | Done |
| HTTP API and web UI with citation viewer | Done |
| Accounts and sign-in (the service currently acts as one built-in user) | Planned |
| Evaluation gates and tracing | Planned |
| Cloud deployment (Terraform) | Planned |

## Design

- **One database.** PostgreSQL holds documents, chunks, dense vectors (pgvector, HNSW)
  and the full-text index, so hybrid search is a single SQL query and deleting a
  document removes everything derived from it in one transaction.
- **Citations are data, not decoration.** A citation is a row linking an answer to a
  chunk, a page and a bounding box, with a support score from a verification pass.
- **Figures and tables are first-class chunks.** Each carries its cropped image and a
  generated description, and is retrieved alongside text.
- **Models are pluggable.** The pipeline talks to a provider interface; hosted and
  local models are a configuration choice.

See [docs/architecture.md](docs/architecture.md) for the data model and request flow.

## Quick start

Requires Docker.

```bash
cp .env.example .env
docker compose up --build -d
curl http://localhost:8000/readyz
```

This starts PostgreSQL with pgvector, applies migrations, serves the API and the web UI
on port 8000 and runs an ingestion worker. Open `http://localhost:8000` for the UI;
interactive API docs are at `http://localhost:8000/docs`.

There is no sign-in yet: every request acts as one built-in user, so keep the service
on a trusted network. Compose publishes it on `127.0.0.1` only.

### Use the web UI

1. Create a collection and upload PDF files (button or drag and drop). Each document
   shows its ingestion state and becomes searchable when it turns ready.
2. Ask a question. This needs `INTELLIQUERY_CHAT_MODEL` set in `.env`; without it the
   UI still manages documents and says what is missing.
3. Click a citation number in the answer to open the source: the page is shown with
   the cited region outlined, next to the passage that was quoted.

Each answer is labelled *Verified*, *Unverified* (the verification pass did not
complete) or *Not in the documents*, and can be rated.

### Ingest documents

Ingestion needs an embedding model that returns 1024-dimensional vectors, served over
any OpenAI-compatible endpoint. With [Ollama](https://ollama.com) on the same machine:

```bash
ollama pull bge-m3
uv sync --extra ingest
uv run alembic upgrade head
uv run intelliquery ingest path/to/manual.pdf --collection manuals
uv run intelliquery worker --drain
uv run intelliquery status
```

To have figures described so they can be retrieved by what they show, set
`INTELLIQUERY_VISION_MODEL` to a vision-capable model. Hosted providers work the same
way: point the `*_BASE_URL` settings at them and set the API keys.

### Ask questions

Set `INTELLIQUERY_CHAT_MODEL` to the model that should write answers, then:

```bash
uv run intelliquery ask "What is the upper alarm limit?" --collection manuals
```

The answer comes back with numbered sources (file, page, section). Sentences the model
could not tie to a source, or that a second verification pass found unsupported, are
removed; if nothing is left, the reply says the documents do not contain the answer.
Pass `--conversation <id>` to ask a follow-up.

### HTTP API

Everything the UI does goes through the JSON API under `/api`:

| Request | Purpose |
|---|---|
| `GET /api/info` | Version and configured models |
| `GET`, `POST /api/collections` · `DELETE /api/collections/{id}` | Manage collections |
| `GET`, `POST /api/collections/{id}/documents` | List documents; upload a PDF (multipart field `file`) |
| `GET`, `DELETE /api/documents/{id}` | Ingestion state; delete a document and everything derived from it |
| `POST /api/collections/{id}/ask` | Ask a question; pass `conversation_id` for a follow-up |
| `GET /api/collections/{id}/conversations` | Conversations in a collection |
| `GET`, `DELETE /api/conversations/{id}` | A conversation with its answers and citations |
| `PUT`, `DELETE /api/messages/{id}/feedback` | Rate an answer |
| `GET /api/documents/{id}/pages/{n}/image` · `GET /api/chunks/{id}/image` | Page render and figure/table crop for the citation viewer |

```bash
curl -X POST http://localhost:8000/api/collections/$COLLECTION/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the upper alarm limit?"}'
```

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # install
docker compose up -d postgres    # database only
uv run alembic upgrade head      # apply migrations
uv run uvicorn intelliquery.api.app:create_app --factory --reload
```

The web UI is a React + TypeScript app in [web/](web) (requires Node.js 22+). During
development it runs on its own server with hot reload and forwards `/api` to the API on
port 8000:

```bash
cd web
npm ci
npm run dev
```

`make web` builds it into `web/dist`; set `INTELLIQUERY_WEB_DIR=web/dist` to have a
locally run API serve that build, as the container image does.

| Task | Command |
|---|---|
| Lint and type-check | `make lint` |
| Web UI: type-check, test, build | `make web` |
| Unit tests | `uv run pytest` |
| Unit + integration tests | set `INTELLIQUERY_DATABASE_URL`, then `uv run pytest` |
| New migration | `make revision m="describe change"` |

Configuration is read from environment variables prefixed `INTELLIQUERY_`;
[.env.example](.env.example) lists them.

## Continuous integration

Every push and pull request runs formatting, lint and strict type checks; applies the
migrations to a real PostgreSQL + pgvector, verifies that models and migrations agree
and that migrations reverse cleanly; runs the test suite; type-checks, tests and builds the
web UI; and builds the container images. Pushes to `main` and version tags publish the image to GitHub Container Registry.

## License

[Apache 2.0](LICENSE)
