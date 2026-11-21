"""VOZ-N0-31: only XPAND operators (``is_superuser``) write the organization's
workflow configuration defaults and Langfuse credentials (VOZ-AC-B0-29), and
they do it on an explicit target organization (``organization_id``).

No database: the user dependency is overridden and the reads, writes and the
organization lookup are stubbed, so the gate itself is what runs.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from api.routes.organization import router
from api.services.auth.depends import get_user
from api.services.configuration.masking import mask_key

ORG_ID = 11  # the caller's selected organization
TARGET_ID = 42  # the organization the operator names
USER_ID = 7
INT32_MAX = 2**31 - 1
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
READS = [DEFAULTS, LANGFUSE]
# The target organization's stored Langfuse document: the body's own values,
# under another project, so a masked key sent back must be restored from it.
STORED_LANGFUSE = {**LANGFUSE_BODY, "project_id": "cm-stored"}

# organization_id values that are not a plain ASCII integer.
MALFORMED = ["", "abc", "4_2", " 42", "+42", "-1", "4.2", "٤٢"]
# Integers that name no organization, in range or not.
UNKNOWN = ["999", "0", str(INT32_MAX + 1), "9" * 30, "9" * 5000]


def _client(*, is_superuser: bool, selected_organization_id=ORG_ID) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_user] = lambda: SimpleNamespace(
        id=USER_ID,
        selected_organization_id=selected_organization_id,
        is_superuser=is_superuser,
    )
    return TestClient(app)


def _params(organization_id):
    return None if organization_id is None else {"organization_id": organization_id}


async def _organization_by_id(organization_id: int):
    # Postgres refuses a parameter outside int32, as asyncpg does.
    if not 0 <= organization_id <= INT32_MAX:
        raise OverflowError("value out of int32 range")
    if organization_id in (ORG_ID, TARGET_ID):
        return SimpleNamespace(id=organization_id)
    return None


async def _langfuse_document(organization_id: int, key: str):
    if organization_id == TARGET_ID:
        return SimpleNamespace(value=dict(STORED_LANGFUSE))
    return None


@pytest.fixture
def stubs():
    """Every read and write the routes can make, stubbed and inspectable."""
    db = MagicMock()
    db.get_configuration = AsyncMock(side_effect=_langfuse_document)
    db.upsert_configuration = AsyncMock()
    db.delete_configuration = AsyncMock(return_value=True)
    auth_db = MagicMock()
    auth_db.get_organization_by_id = AsyncMock(side_effect=_organization_by_id)
    upsert_defaults = AsyncMock(return_value={"max_call_duration": 600})
    read_defaults = AsyncMock(return_value={"max_call_duration": 600})
    sync = MagicMock()
    sync.broadcast = AsyncMock()
    with (
        patch("api.routes.organization.db_client", db),
        patch("api.services.auth.depends.db_client", auth_db),
        patch(
            "api.routes.organization.upsert_organization_workflow_configuration_defaults",
            upsert_defaults,
        ),
        patch(
            "api.routes.organization.get_organization_workflow_configuration_defaults",
            read_defaults,
        ),
        patch("api.routes.organization.get_worker_sync_manager", return_value=sync),
    ):
        yield SimpleNamespace(
            db=db,
            upsert_defaults=upsert_defaults,
            read_defaults=read_defaults,
            upsert_configuration=db.upsert_configuration,
            delete_configuration=db.delete_configuration,
            broadcast=sync.broadcast,
        )


def _assert_nothing_written(stubs):
    stubs.upsert_defaults.assert_not_called()
    stubs.upsert_configuration.assert_not_called()
    stubs.delete_configuration.assert_not_called()
    stubs.broadcast.assert_not_called()


def _assert_b028(response, *, status, code, method, path):
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code
    # VOZ-AC-B6-53: where names the refused route.
    assert detail["where"] == f"{method.upper()} {path}"
    assert detail["reason"] and detail["hint"]
    return detail


def _capture_logs():
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), format="{message}")
    return lines, sink_id


@pytest.mark.parametrize("organization_id", [None, str(TARGET_ID), "abc", ""])
@pytest.mark.parametrize(("method", "path", "body", "write"), PROTECTED)
def test_non_superuser_gets_403_b028_and_nothing_is_written(
    stubs, method, path, body, write, organization_id
):
    response = _client(is_superuser=False).request(
        method, path, json=body, params=_params(organization_id)
    )

    detail = _assert_b028(
        response, status=403, code="privacy_knob_forbidden", method=method, path=path
    )
    assert "XPAND operator" in detail["hint"]
    _assert_nothing_written(stubs)


@pytest.mark.parametrize("organization_id", [None, *MALFORMED])
@pytest.mark.parametrize(("method", "path", "body", "write"), PROTECTED)
def test_superuser_without_a_target_organization_gets_422(
    stubs, method, path, body, write, organization_id
):
    response = _client(is_superuser=True).request(
        method, path, json=body, params=_params(organization_id)
    )

    detail = _assert_b028(
        response,
        status=422,
        code="privacy_target_organization_required",
        method=method,
        path=path,
    )
    assert "organization_id" in detail["hint"]
    _assert_nothing_written(stubs)


@pytest.mark.parametrize("organization_id", UNKNOWN)
@pytest.mark.parametrize(("method", "path", "body", "write"), PROTECTED)
def test_unknown_target_organization_gets_404(
    stubs, method, path, body, write, organization_id
):
    response = _client(is_superuser=True).request(
        method, path, json=body, params=_params(organization_id)
    )

    _assert_b028(
        response, status=404, code="organization_not_found", method=method, path=path
    )
    _assert_nothing_written(stubs)


@pytest.mark.parametrize(("method", "path", "body", "write"), PROTECTED)
def test_operator_without_a_selected_organization_writes_the_target(
    stubs, method, path, body, write
):
    response = _client(is_superuser=True, selected_organization_id=None).request(
        method, path, json=body, params=_params(str(TARGET_ID))
    )

    assert response.status_code == 200, response.text
    getattr(stubs, write).assert_awaited_once()
    assert getattr(stubs, write).await_args.args[0] == TARGET_ID


def test_defaults_put_passes_the_target_and_the_operator(stubs):
    response = _client(is_superuser=True).put(
        DEFAULTS, json={"max_call_duration": 600}, params=_params(str(TARGET_ID))
    )

    assert response.status_code == 200, response.text
    assert stubs.upsert_defaults.await_args.args == (TARGET_ID,)
    assert stubs.upsert_defaults.await_args.kwargs["user_id"] == USER_ID


def test_langfuse_save_carries_the_target_everywhere(stubs):
    # Masked keys, as the GET returns them: the pre-read must find the target's
    # stored document to restore them.
    body = {
        **LANGFUSE_BODY,
        "public_key": mask_key(STORED_LANGFUSE["public_key"]),
        "secret_key": mask_key(STORED_LANGFUSE["secret_key"]),
    }
    lines, sink_id = _capture_logs()
    try:
        response = _client(is_superuser=True).post(
            LANGFUSE, json=body, params=_params(str(TARGET_ID))
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 200, response.text
    assert stubs.db.get_configuration.await_args.args[0] == TARGET_ID
    organization_id, _, stored = stubs.upsert_configuration.await_args.args
    assert organization_id == TARGET_ID
    assert stored["public_key"] == STORED_LANGFUSE["public_key"]
    assert stored["secret_key"] == STORED_LANGFUSE["secret_key"]
    assert stubs.broadcast.await_args.kwargs["org_id"] == TARGET_ID
    assert (
        f"organization {TARGET_ID} langfuse credentials saved by user {USER_ID}"
        in "".join(lines)
    )
    assert all(ORG_ID not in call.args for call in stubs.db.mock_calls)


def test_langfuse_delete_carries_the_target_everywhere(stubs):
    lines, sink_id = _capture_logs()
    try:
        response = _client(is_superuser=True).delete(
            LANGFUSE, params=_params(str(TARGET_ID))
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 200, response.text
    assert stubs.delete_configuration.await_args.args[0] == TARGET_ID
    assert stubs.broadcast.await_args.kwargs["org_id"] == TARGET_ID
    assert (
        f"organization {TARGET_ID} langfuse credentials deleted by user {USER_ID}"
        in "".join(lines)
    )
    assert all(ORG_ID not in call.args for call in stubs.db.mock_calls)


def test_non_superuser_reads_their_own_organization_as_today(stubs):
    client = _client(is_superuser=False)

    defaults = client.get(DEFAULTS)
    langfuse = client.get(LANGFUSE)

    assert defaults.status_code == 200, defaults.text
    assert stubs.read_defaults.await_args.args == (ORG_ID,)
    assert langfuse.status_code == 200, langfuse.text
    assert stubs.db.get_configuration.await_args.args[0] == ORG_ID


def test_superuser_reads_the_named_organization(stubs):
    client = _client(is_superuser=True)

    defaults = client.get(DEFAULTS, params=_params(str(TARGET_ID)))
    langfuse = client.get(LANGFUSE, params=_params(str(TARGET_ID)))

    assert defaults.status_code == 200, defaults.text
    assert stubs.read_defaults.await_args.args == (TARGET_ID,)
    assert langfuse.status_code == 200, langfuse.text
    assert stubs.db.get_configuration.await_args.args[0] == TARGET_ID
    assert langfuse.json()["configured"] is True
    assert langfuse.json()["project_id"] == STORED_LANGFUSE["project_id"]
    assert langfuse.json()["secret_key"] == mask_key(STORED_LANGFUSE["secret_key"])


@pytest.mark.parametrize("organization_id", [str(TARGET_ID), "abc", ""])
@pytest.mark.parametrize("path", READS)
def test_non_superuser_naming_an_organization_on_a_read_gets_403(
    stubs, path, organization_id
):
    response = _client(is_superuser=False).get(path, params=_params(organization_id))

    _assert_b028(
        response, status=403, code="privacy_knob_forbidden", method="get", path=path
    )
    stubs.read_defaults.assert_not_called()
    stubs.db.get_configuration.assert_not_called()


@pytest.mark.parametrize(
    ("organization_id", "status", "code"),
    [
        ("", 422, "privacy_target_organization_required"),
        ("abc", 422, "privacy_target_organization_required"),
        ("999", 404, "organization_not_found"),
        (str(INT32_MAX + 1), 404, "organization_not_found"),
    ],
)
@pytest.mark.parametrize("path", READS)
def test_superuser_read_of_a_bad_target_is_refused(
    stubs, path, organization_id, status, code
):
    response = _client(is_superuser=True).get(path, params=_params(organization_id))

    _assert_b028(response, status=status, code=code, method="get", path=path)
    stubs.read_defaults.assert_not_called()
    stubs.db.get_configuration.assert_not_called()


@pytest.mark.parametrize(
    ("method", "path", "body", "event"),
    [
        ("post", LANGFUSE, LANGFUSE_BODY, "saved"),
        ("delete", LANGFUSE, None, "deleted"),
    ],
)
def test_langfuse_audit_line_survives_a_failed_broadcast(
    stubs, method, path, body, event
):
    """The database write is committed before the broadcast, so the audit line
    must be too: a Redis failure may not leave a change nobody signed."""
    stubs.broadcast.side_effect = RuntimeError("redis unreachable")
    lines, sink_id = _capture_logs()
    try:
        with pytest.raises(RuntimeError):
            _client(is_superuser=True).request(
                method, path, json=body, params=_params(str(TARGET_ID))
            )
    finally:
        logger.remove(sink_id)

    assert (
        f"organization {TARGET_ID} langfuse credentials {event} by user {USER_ID}"
        in "".join(lines)
    )
    assert "sk-lf-1" not in "".join(lines)
