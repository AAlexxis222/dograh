"""VOZ-N0-31: only XPAND operators (``is_superuser``) write the organization's
workflow configuration defaults and Langfuse credentials (VOZ-AC-B0-29), and
they do it on a target organization named in the path
(``/organizations/{organization_id}/...``). The routes without an organization
in the path never write: the previous version served them and ignored
``organization_id``, so a write there would land on the operator's own
organization.

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
# The operator routes: the target organization is a path segment.
TARGETED_DEFAULTS = "/organizations/{}/workflow-configuration-defaults"
TARGETED_LANGFUSE = "/organizations/{}/langfuse-credentials"
LANGFUSE_BODY = {
    "host": "https://cloud.langfuse.com",
    "public_key": "pk-lf-1",
    "secret_key": "sk-lf-1",
    "project_id": "cm1",
}

# (method, path template, json body, name of the write the route must not reach)
PROTECTED = [
    ("put", TARGETED_DEFAULTS, {"max_call_duration": 600}, "upsert_defaults"),
    ("post", TARGETED_LANGFUSE, LANGFUSE_BODY, "upsert_configuration"),
    ("delete", TARGETED_LANGFUSE, None, "delete_configuration"),
]
# The same writes on the routes without an organization in the path.
UNTARGETED = [
    ("put", DEFAULTS, {"max_call_duration": 600}),
    ("post", LANGFUSE, LANGFUSE_BODY),
    ("delete", LANGFUSE, None),
]
TARGETED_READS = [TARGETED_DEFAULTS, TARGETED_LANGFUSE]
# The target organization's stored Langfuse document: the body's own values,
# under another project, so a masked key sent back must be restored from it.
STORED_LANGFUSE = {**LANGFUSE_BODY, "project_id": "cm-stored"}

# Organization ids that are not a plain ASCII integer. An empty segment
# (/organizations//...) matches no route at all, so it is not among them.
MALFORMED = ["abc", "4_2", " 42", "+42", "-1", "4.2", "٤٢"]
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


def _targeted_route(template: str, method: str) -> str:
    """The method and path an operator is pointed to, as the hint names it."""
    return f"{method.upper()} {template.format('{organization_id}')}"


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
        patch("api.services.auth.privacy_operator.db_client", auth_db),
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


@pytest.mark.parametrize("organization_id", [str(TARGET_ID), str(ORG_ID), "abc"])
@pytest.mark.parametrize(("method", "template", "body", "write"), PROTECTED)
def test_non_superuser_gets_403_b028_and_nothing_is_written(
    stubs, method, template, body, write, organization_id
):
    path = template.format(organization_id)
    response = _client(is_superuser=False).request(method, path, json=body)

    detail = _assert_b028(
        response, status=403, code="privacy_knob_forbidden", method=method, path=path
    )
    assert "XPAND operator" in detail["hint"]
    _assert_nothing_written(stubs)


@pytest.mark.parametrize("organization_id", MALFORMED)
@pytest.mark.parametrize(("method", "template", "body", "write"), PROTECTED)
def test_superuser_with_a_malformed_target_organization_gets_422(
    stubs, method, template, body, write, organization_id
):
    path = template.format(organization_id)
    response = _client(is_superuser=True).request(method, path, json=body)

    detail = _assert_b028(
        response,
        status=422,
        code="privacy_target_organization_required",
        method=method,
        path=path,
    )
    assert "{organization_id}" in detail["hint"]
    _assert_nothing_written(stubs)


@pytest.mark.parametrize("organization_id", [None, str(TARGET_ID), "abc"])
@pytest.mark.parametrize(("method", "path", "body"), UNTARGETED)
def test_non_superuser_write_without_an_organization_in_the_path_gets_403(
    stubs, method, path, body, organization_id
):
    response = _client(is_superuser=False).request(
        method, path, json=body, params=_params(organization_id)
    )

    detail = _assert_b028(
        response, status=403, code="privacy_knob_forbidden", method=method, path=path
    )
    assert "XPAND operator" in detail["hint"]
    _assert_nothing_written(stubs)


@pytest.mark.parametrize("organization_id", [None, str(TARGET_ID), "abc"])
@pytest.mark.parametrize(("method", "path", "body"), UNTARGETED)
def test_operator_write_without_an_organization_in_the_path_never_writes(
    stubs, method, path, body, organization_id
):
    """The previous version accepted this exact request and wrote the operator's
    own organization: the query parameter is never read, the write is refused
    and the hint names the route that takes the organization in the path."""
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
    targeted = TARGETED_DEFAULTS if path == DEFAULTS else TARGETED_LANGFUSE
    assert _targeted_route(targeted, method) in detail["hint"]
    _assert_nothing_written(stubs)


@pytest.mark.parametrize("organization_id", UNKNOWN)
@pytest.mark.parametrize(("method", "template", "body", "write"), PROTECTED)
def test_unknown_target_organization_gets_404(
    stubs, method, template, body, write, organization_id
):
    path = template.format(organization_id)
    response = _client(is_superuser=True).request(method, path, json=body)

    _assert_b028(
        response, status=404, code="organization_not_found", method=method, path=path
    )
    _assert_nothing_written(stubs)


@pytest.mark.parametrize(("method", "template", "body", "write"), PROTECTED)
def test_operator_without_a_selected_organization_writes_the_target(
    stubs, method, template, body, write
):
    response = _client(is_superuser=True, selected_organization_id=None).request(
        method, template.format(TARGET_ID), json=body
    )

    assert response.status_code == 200, response.text
    getattr(stubs, write).assert_awaited_once()
    assert getattr(stubs, write).await_args.args[0] == TARGET_ID


def test_defaults_put_passes_the_target_and_the_operator(stubs):
    response = _client(is_superuser=True).put(
        TARGETED_DEFAULTS.format(TARGET_ID), json={"max_call_duration": 600}
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
            TARGETED_LANGFUSE.format(TARGET_ID), json=body
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
            TARGETED_LANGFUSE.format(TARGET_ID)
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


@pytest.mark.parametrize("is_superuser", [False, True])
@pytest.mark.parametrize("organization_id", [None, str(TARGET_ID), "abc"])
def test_reads_without_an_organization_in_the_path_are_the_callers_own(
    stubs, is_superuser, organization_id
):
    """As at base: the selected organization, and organization_id is ignored."""
    client = _client(is_superuser=is_superuser)

    defaults = client.get(DEFAULTS, params=_params(organization_id))
    langfuse = client.get(LANGFUSE, params=_params(organization_id))

    assert defaults.status_code == 200, defaults.text
    assert stubs.read_defaults.await_args.args == (ORG_ID,)
    assert langfuse.status_code == 200, langfuse.text
    assert stubs.db.get_configuration.await_args.args[0] == ORG_ID


def test_superuser_reads_the_named_organization(stubs):
    client = _client(is_superuser=True)

    defaults = client.get(TARGETED_DEFAULTS.format(TARGET_ID))
    langfuse = client.get(TARGETED_LANGFUSE.format(TARGET_ID))

    assert defaults.status_code == 200, defaults.text
    assert stubs.read_defaults.await_args.args == (TARGET_ID,)
    assert langfuse.status_code == 200, langfuse.text
    assert stubs.db.get_configuration.await_args.args[0] == TARGET_ID
    assert langfuse.json()["configured"] is True
    assert langfuse.json()["project_id"] == STORED_LANGFUSE["project_id"]
    assert langfuse.json()["secret_key"] == mask_key(STORED_LANGFUSE["secret_key"])


@pytest.mark.parametrize("organization_id", [str(TARGET_ID), str(ORG_ID), "abc"])
@pytest.mark.parametrize("template", TARGETED_READS)
def test_non_superuser_naming_an_organization_on_a_read_gets_403(
    stubs, template, organization_id
):
    path = template.format(organization_id)
    response = _client(is_superuser=False).get(path)

    _assert_b028(
        response, status=403, code="privacy_knob_forbidden", method="get", path=path
    )
    stubs.read_defaults.assert_not_called()
    stubs.db.get_configuration.assert_not_called()


@pytest.mark.parametrize(
    ("organization_id", "status", "code"),
    [
        ("abc", 422, "privacy_target_organization_required"),
        ("999", 404, "organization_not_found"),
        (str(INT32_MAX + 1), 404, "organization_not_found"),
    ],
)
@pytest.mark.parametrize("template", TARGETED_READS)
def test_superuser_read_of_a_bad_target_is_refused(
    stubs, template, organization_id, status, code
):
    path = template.format(organization_id)
    response = _client(is_superuser=True).get(path)

    _assert_b028(response, status=status, code=code, method="get", path=path)
    stubs.read_defaults.assert_not_called()
    stubs.db.get_configuration.assert_not_called()


@pytest.mark.parametrize(
    ("method", "body", "event"),
    [("post", LANGFUSE_BODY, "saved"), ("delete", None, "deleted")],
)
def test_langfuse_audit_line_survives_a_failed_broadcast(stubs, method, body, event):
    """The database write is committed before the broadcast, so the audit line
    must be too: a Redis failure may not leave a change nobody signed."""
    stubs.broadcast.side_effect = RuntimeError("redis unreachable")
    lines, sink_id = _capture_logs()
    try:
        with pytest.raises(RuntimeError):
            _client(is_superuser=True).request(
                method, TARGETED_LANGFUSE.format(TARGET_ID), json=body
            )
    finally:
        logger.remove(sink_id)

    assert (
        f"organization {TARGET_ID} langfuse credentials {event} by user {USER_ID}"
        in "".join(lines)
    )
    assert "sk-lf-1" not in "".join(lines)


@pytest.mark.parametrize(
    ("path", "method"),
    [
        (DEFAULTS, "put"),
        (LANGFUSE, "post"),
        (TARGETED_DEFAULTS.format("{organization_id}"), "put"),
        (TARGETED_LANGFUSE.format("{organization_id}"), "post"),
    ],
)
def test_routes_with_a_body_keep_the_framework_validation_422(path, method):
    """Body validation errors carry FastAPI's list detail: the published 422
    keeps that schema, and the B0-28 refusal is only described next to it."""
    app = FastAPI()
    app.include_router(router)

    response = app.openapi()["paths"][path][method]["responses"]["422"]

    schema = response["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/HTTPValidationError"}
    assert "privacy_target_organization_required" in response["description"]
