"""Application settings, validated once at startup."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dimension of the embedding column. Changing it requires a migration and a re-embed
# of every chunk, so it is a schema constant rather than a runtime setting.
EMBEDDING_DIMENSIONS = 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INTELLIQUERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    database_url: PostgresDsn = Field(
        default=PostgresDsn(
            "postgresql+asyncpg://intelliquery:intelliquery@localhost:5432/intelliquery"
        )
    )
    database_pool_size: int = Field(default=10, ge=1)
    database_max_overflow: int = Field(default=10, ge=0)

    # Object storage for source files, page renders and figure crops.
    storage_dir: Path = Path("data/storage")

    # Model endpoints speak the OpenAI-compatible HTTP API, which covers hosted
    # providers as well as local servers such as Ollama. Defaults target local Ollama.
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "bge-m3"
    # Send the `dimensions` parameter, for models with configurable output size.
    embedding_request_dimensions: bool = False
    embedding_batch_size: int = Field(default=32, ge=1)

    # Figure description is skipped when no vision model is configured.
    vision_base_url: str = "http://localhost:11434/v1"
    vision_api_key: SecretStr | None = None
    vision_model: str | None = None
    vision_concurrency: int = Field(default=4, ge=1)

    model_timeout_seconds: float = Field(default=120.0, gt=0)
    model_max_retries: int = Field(default=3, ge=0)

    ingest_max_upload_mb: int = Field(default=100, ge=1)
    ingest_max_chunk_tokens: int = Field(default=350, ge=50)
    ingest_image_scale: float = Field(default=2.0, gt=0)
    ingest_ocr: bool = False
    ingest_max_attempts: int = Field(default=3, ge=1)
    # A running job older than this is assumed to belong to a dead worker.
    ingest_job_timeout_seconds: int = Field(default=1800, ge=60)
    worker_poll_seconds: float = Field(default=2.0, gt=0)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
