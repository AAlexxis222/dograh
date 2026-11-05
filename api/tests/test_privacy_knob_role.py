"""VOZ-N0-31: only XPAND operators (``is_superuser``) write the organization's
workflow configuration defaults and Langfuse credentials (VOZ-AC-B0-29).

No database: the user dependency is overridden and the writes are stubbed, so
the guard itself (``get_privacy_operator``) is what runs.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from api.routes.organization import router
from api.services.auth.depends import get_user_with_selected_organization

ORG_ID = 11
USER_ID = 7
DEFAULTS = "/organizations/workflow-configuration-defaults"
LANGFUSE = "/organizations/langfuse-credentials"
LANGFUSE_BODY = {
    "host": "https://cloud.langfuse.com",
    "public_key": "pk-lf-1",
    "secret_key": "sk-lf-1",
    "project_id": "cm1",
}

# (method, path, json body, name of the write the route must not reach)
PROTECTED = [
    ("put", DEFAULTS, {"max_call_duration": 600}, "upsert_defaults"),
    ("post", LANGFUSE, LANGFUSE_BODY, "upsert_configuration"),
    ("delete", LANGFUSE, None, "delete_configuration"),
]


def _client(*, is_superuser: bool) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_user_with_selected_organization] = lambda: (
        SimpleNamespace(
            id=USER_ID, selected_organization_id=ORG_ID, is_superuser=is_superuser
        )
    )
    return TestClient(app)


@pytest.fixture
def writes():
    """Every write the three routes can make, stubbed and inspectable."""
    db = MagicMock()
    db.get_configuration = AsyncMock(return_value=None)
    db.upsert_configuration = AsyncMock()
    db.delete_configuration = AsyncMock(return_value=True)
    upsert_defaults = AsyncMock(return_value={"max_call_duration": 600})
    sync = MagicMock()
    sync.broadcast = AsyncMock()
    with (
        patch("api.routes.organization.db_client", db),
        patch(
            "api.routes.organization.upsert_organization_workflow_configuration_defaults",
            upsert_defaults,
        ),
        patch("api.routes.organization.get_worker_sync_manager", return_value=sync),
    ):
        yield SimpleNamespace(
            upsert_defaults=upsert_defaults,
            upsert_configuration=db.upsert_configuration,
            delete_configuration=db.delete_configuration,
            broadcast=sync.broadcast,
        )


@pytest.mark.parametrize(("method", "path", "body", "write"), PROTECTED)
def test_non_superuser_gets_403_b028_and_nothing_is_written(
    writes, method, path, body, write
):
    response = _client(is_superuser=False).request(method, path, json=body)

    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "privacy_knob_forbidden"
    assert detail["where"] and detail["reason"]
    assert "XPAND operator" in detail["hint"]
    getattr(writes, write).assert_not_called()
    writes.broadcast.assert_not_called()


@pytest.mark.parametrize(("method", "path", "body", "write"), PROTECTED)
def test_superuser_still_reaches_the_write(writes, method, path, body, write):
    response = _client(is_superuser=True).request(method, path, json=body)

    assert response.status_code == 200, response.text
    getattr(writes, write).assert_awaited_once()


def test_non_superuser_can_still_read_the_defaults(writes):
    with patch(
        "api.routes.organization.get_organization_workflow_configuration_defaults",
        AsyncMock(return_value={"max_call_duration": 600}),
    ):
        response = _client(is_superuser=False).get(DEFAULTS)

    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    ("method", "path", "body", "event"),
    [
        ("post", LANGFUSE, LANGFUSE_BODY, "saved"),
        ("delete", LANGFUSE, None, "deleted"),
    ],
)
def test_langfuse_writes_log_the_operator_user_id(writes, method, path, body, event):
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), format="{message}")
    try:
        response = _client(is_superuser=True).request(method, path, json=body)
    finally:
        logger.remove(sink_id)

    assert response.status_code == 200, response.text
    assert (
        f"organization {ORG_ID} langfuse credentials {event} by user {USER_ID}"
        in "".join(lines)
    )
    assert "sk-lf-1" not in "".join(lines)
