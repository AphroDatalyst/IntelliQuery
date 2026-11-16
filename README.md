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
| Chat API and web UI with citation viewer | Planned |
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

This starts PostgreSQL with pgvector, applies migrations, serves the API on port 8000
and runs an ingestion worker. Interactive API docs are at `http://localhost:8000/docs`.

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

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # install
docker compose up -d postgres    # database only
uv run alembic upgrade head      # apply migrations
uv run uvicorn intelliquery.api.app:create_app --factory --reload
```

| Task | Command |
|---|---|
| Lint and type-check | `make lint` |
| Unit tests | `uv run pytest` |
| Unit + integration tests | set `INTELLIQUERY_DATABASE_URL`, then `uv run pytest` |
| New migration | `make revision m="describe change"` |

Configuration is read from environment variables prefixed `INTELLIQUERY_`;
[.env.example](.env.example) lists them.

## Continuous integration

Every push and pull request runs formatting, lint and strict type checks; applies the
migrations to a real PostgreSQL + pgvector, verifies that models and migrations agree
and that migrations reverse cleanly; runs the test suite; and builds the container
image. Pushes to `main` and version tags publish the image to GitHub Container Registry.

## License

[Apache 2.0](LICENSE)
