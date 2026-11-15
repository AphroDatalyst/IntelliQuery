import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from intelliquery.config import Settings
from intelliquery.db.session import create_engine


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(environment="test")


@pytest.fixture(scope="session")
async def engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    """Engine for a migrated database; integration tests are skipped without one."""
    if not os.environ.get("INTELLIQUERY_DATABASE_URL"):
        pytest.skip("INTELLIQUERY_DATABASE_URL is not set")
    engine = create_engine(settings)
    yield engine
    await engine.dispose()
