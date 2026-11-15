"""Application settings, validated once at startup."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn
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

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
