"""Liveness and readiness probes."""

import logging

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from intelliquery import __version__
from intelliquery.db.session import check_database

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


class Liveness(BaseModel):
    status: str
    version: str


class Readiness(BaseModel):
    status: str
    checks: dict[str, str]


@router.get("/healthz")
async def healthz() -> Liveness:
    """The process is up. Does not touch dependencies."""
    return Liveness(status="ok", version=__version__)


@router.get("/readyz", responses={503: {"model": Readiness}})
async def readyz(request: Request, response: Response) -> Readiness:
    """The service can do work: the database is reachable and pgvector is installed."""
    try:
        versions = await check_database(request.app.state.engine)
    except Exception:
        logger.exception("readiness check failed")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Readiness(status="unavailable", checks={"database": "unreachable"})
    return Readiness(status="ok", checks=versions)
