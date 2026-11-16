import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from intelliquery.config import Settings
from intelliquery.db.session import create_engine, create_session_factory
from intelliquery.storage import LocalStorage


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


@pytest.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory over an emptied database.

    Tests that commit need a clean slate, so this truncates every table. It refuses
    to run against anything but a database whose name ends in `_test`.
    """
    name = engine.url.database or ""
    if not name.endswith("_test"):
        pytest.fail(f"refusing to truncate {name!r}: test database names must end in '_test'")
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE users, eval_runs CASCADE"))
    return create_session_factory(engine)


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "storage")
