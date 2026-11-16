# syntax=docker/dockerfile:1.7
# Ingestion worker: the API image plus the document parser and PyTorch (CPU).

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ingest --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ingest --no-editable


FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    INTELLIQUERY_LOG_JSON=true \
    INTELLIQUERY_STORAGE_DIR=/var/lib/intelliquery/storage \
    # Parser model weights are downloaded on first use; keep them on a volume.
    HF_HOME=/var/lib/intelliquery/models

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app \
    && mkdir -p /var/lib/intelliquery/storage /var/lib/intelliquery/models \
    && chown -R app:app /var/lib/intelliquery
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
USER app

CMD ["intelliquery", "worker"]
