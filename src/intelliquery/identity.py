"""Who is acting.

Accounts are not implemented yet: everything is owned by one built-in user. Callers
ask this module for the acting user and never look the account up themselves, so
real authentication can replace it without touching them.
"""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from intelliquery.db import models

LOCAL_USER_EMAIL = "local@intelliquery.invalid"


async def get_local_user(session: AsyncSession) -> models.User:
    """Return the built-in user, creating it on first use."""
    user = await session.scalar(select(models.User).where(models.User.email == LOCAL_USER_EMAIL))
    if user is None:
        # Two first requests can race; the unique e-mail makes the loser a no-op.
        # "!" is not a valid password hash, so this account can never be logged in to.
        await session.execute(
            insert(models.User)
            .values(email=LOCAL_USER_EMAIL, hashed_password="!")  # noqa: S106
            .on_conflict_do_nothing(index_elements=[models.User.email])
        )
        user = await session.scalar(
            select(models.User).where(models.User.email == LOCAL_USER_EMAIL)
        )
        if user is None:  # pragma: no cover - the row was just inserted
            raise RuntimeError("could not create the local user")
    return user
