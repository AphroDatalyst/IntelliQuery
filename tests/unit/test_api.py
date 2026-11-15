import json
import logging

import pytest
from fastapi.testclient import TestClient

from intelliquery import __version__
from intelliquery.api.app import REQUEST_ID_HEADER, create_app
from intelliquery.config import Settings
from intelliquery.logging import JsonFormatter, request_id_var

# Nothing listens here, so readiness must fail fast rather than hang.
UNREACHABLE_DB = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none"


@pytest.fixture
def client() -> TestClient:
    settings = Settings(environment="test", database_url=UNREACHABLE_DB)
    return TestClient(create_app(settings))


def test_healthz_does_not_need_the_database(client: TestClient) -> None:
    with client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_readyz_reports_503_when_database_is_down(client: TestClient) -> None:
    with client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "checks": {"database": "unreachable"}}


def test_request_id_is_generated_and_echoed(client: TestClient) -> None:
    with client:
        generated = client.get("/healthz").headers[REQUEST_ID_HEADER]
        echoed = client.get("/healthz", headers={REQUEST_ID_HEADER: "abc123"})
    assert len(generated) == 32
    assert echoed.headers[REQUEST_ID_HEADER] == "abc123"


def test_docs_are_disabled_in_production() -> None:
    settings = Settings(environment="production", database_url=UNREACHABLE_DB)
    with TestClient(create_app(settings)) as client:
        assert client.get("/docs").status_code == 404


def test_json_formatter_includes_request_id_and_extras() -> None:
    record = logging.makeLogRecord({"name": "t", "levelname": "INFO", "msg": "hello %s"})
    record.args = ("world",)
    record.status = 200
    token = request_id_var.set("rid-1")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "rid-1"
    assert payload["status"] == 200
