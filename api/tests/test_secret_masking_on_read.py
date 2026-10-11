"""Registered secrets never leave in clear and a masked value sent back keeps the stored one (VOZ-N0-24b).

Covers the webhook node's ``custom_headers`` and a tool's static headers (mask on GET, merge-back on save), the
fixed-width mask of VOZ-AC-B6-19 (show-once, minimal) and the run detail showing references.
"""

import copy
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.routes.tool import UpdateToolRequest
from api.routes.tool import update_tool as call_update_tool_route
from api.services.configuration.credential_refs import REDACTED, credential_ref
from api.services.configuration.masking import (
    MaskedSecretUnmatched,
    is_mask_of,
    mask_key,
    mask_tool_definition,
    mask_workflow_definition,
    merge_tool_headers,
    merge_workflow_api_keys,
)
from api.services.configuration.secrets_registry import mask_secrets
from api.services.tool_management import build_tool_response

CANARY = "canary-header-secret-0003"


# --- VOZ-AC-B6-19: the mask never encodes the secret's length ------------------------------------------------------


def test_the_mask_has_a_fixed_width_and_shows_only_the_last_four():
    assert (
        mask_key("short-cdef")
        == mask_key("a-much-longer-secret-value-cdef")
        == "****cdef"
    )
    assert mask_key("abcd") == "****"  # nothing shown when the secret is that short
    assert mask_key("") == ""


def test_a_mask_in_the_previous_shape_is_still_recognised():
    real = "sk-1234567890abcdef"
    assert is_mask_of("*" * (len(real) - 4) + "cdef", real)
    assert is_mask_of(mask_key(real), real)
    assert not is_mask_of("****beef", real)


# --- webhook node custom headers --------------------------------------------------------------------------------


def _webhook(node_id="hook", headers=None):
    return {
        "id": node_id,
        "type": "webhook",
        "data": {
            "name": "Hook",
            "endpoint_url": "https://x",
            "custom_headers": headers,
        },
    }


def _graph(*nodes):
    return {"nodes": list(nodes), "edges": []}


STORED = _graph(
    _webhook(
        headers=[
            {"key": "Authorization", "value": f"Bearer {CANARY}"},
            {"key": "X-Tenant", "value": "tenant-42"},
        ]
    )
)


def _masked_headers():
    return copy.deepcopy(
        mask_workflow_definition(STORED)["nodes"][0]["data"]["custom_headers"]
    )


def test_custom_header_values_are_masked_on_read():
    masked = mask_workflow_definition(STORED)
    assert CANARY not in json.dumps(masked)
    headers = masked["nodes"][0]["data"]["custom_headers"]
    assert [h["key"] for h in headers] == ["Authorization", "X-Tenant"]
    assert headers[0]["value"] == mask_key(f"Bearer {CANARY}")
    assert (
        STORED["nodes"][0]["data"]["custom_headers"][0]["value"] == f"Bearer {CANARY}"
    )


def test_a_masked_header_sent_back_keeps_the_stored_value_even_reordered():
    incoming = _graph(_webhook(headers=list(reversed(_masked_headers()))))
    merged = merge_workflow_api_keys(incoming, STORED)
    assert merged["nodes"][0]["data"]["custom_headers"] == [
        {"key": "X-Tenant", "value": "tenant-42"},
        {"key": "Authorization", "value": f"Bearer {CANARY}"},
    ]


def test_an_omitted_header_is_removed_and_a_new_one_kept_next_to_masked_neighbours():
    headers = _masked_headers()[:1] + [{"key": "X-New", "value": "plain-new"}]
    merged = merge_workflow_api_keys(_graph(_webhook(headers=headers)), STORED)
    assert merged["nodes"][0]["data"]["custom_headers"] == [
        {"key": "Authorization", "value": f"Bearer {CANARY}"},
        {"key": "X-New", "value": "plain-new"},
    ]


def test_a_changed_header_value_replaces_the_stored_one():
    headers = _masked_headers()
    headers[0]["value"] = "Bearer rotated"
    merged = merge_workflow_api_keys(_graph(_webhook(headers=headers)), STORED)
    assert merged["nodes"][0]["data"]["custom_headers"][0]["value"] == "Bearer rotated"


@pytest.mark.parametrize(
    "incoming",
    [
        _graph(_webhook(headers=[{"key": "X-Renamed", "value": "****" + CANARY[-4:]}])),
        _graph(_webhook(node_id="other", headers=_masked_headers())),
    ],
    ids=["renamed_header", "other_node"],
)
def test_a_mask_without_a_stored_value_is_refused(incoming):
    with pytest.raises(MaskedSecretUnmatched) as raised:
        merge_workflow_api_keys(incoming, STORED)
    error = raised.value
    assert error.code == "masked_secret_unmatched"
    assert "custom_headers" in error.where and error.reason and error.hint
    assert CANARY not in str(error)


# --- tool static headers ----------------------------------------------------------------------------------------


def _tool_definition(headers, resolver_headers=None):
    config = {"method": "POST", "url": "https://example.com/hook", "headers": headers}
    if resolver_headers is not None:
        config["resolver"] = {
            "url": "https://example.com/route",
            "headers": resolver_headers,
        }
    return {"schema_version": 1, "type": "http_api", "config": config}


TOOL_STORED = _tool_definition(
    {"Authorization": f"Bearer {CANARY}", "X-Tenant": "tenant-42"},
    resolver_headers={"X-Resolver-Key": f"{CANARY}-resolver"},
)


def test_tool_headers_are_masked_in_every_tool_response():
    tool = SimpleNamespace(
        id=1,
        tool_uuid="tu-1",
        name="T",
        description=None,
        category="http_api",
        icon=None,
        icon_color=None,
        status="active",
        definition=copy.deepcopy(TOOL_STORED),
        created_at=datetime.now(UTC),
        updated_at=None,
    )
    body = build_tool_response(tool).model_dump_json()
    assert CANARY not in body
    masked = mask_tool_definition(TOOL_STORED)
    assert masked["config"]["headers"]["Authorization"] == mask_key(f"Bearer {CANARY}")
    assert masked["config"]["resolver"]["headers"]["X-Resolver-Key"] == mask_key(
        f"{CANARY}-resolver"
    )
    assert TOOL_STORED["config"]["headers"]["Authorization"] == f"Bearer {CANARY}"


def test_tool_headers_merge_back_by_name_and_omitted_ones_go():
    incoming = mask_tool_definition(TOOL_STORED)
    del incoming["config"]["headers"]["X-Tenant"]
    incoming["config"]["headers"]["X-New"] = "plain"
    merged = merge_tool_headers(incoming, TOOL_STORED)
    assert merged["config"]["headers"] == {
        "Authorization": f"Bearer {CANARY}",
        "X-New": "plain",
    }
    assert merged["config"]["resolver"]["headers"] == {
        "X-Resolver-Key": f"{CANARY}-resolver"
    }


@pytest.mark.parametrize(
    "stored", [TOOL_STORED, None], ids=["renamed", "no_stored_tool"]
)
def test_a_tool_header_mask_without_a_stored_value_is_refused(stored):
    incoming = _tool_definition({"X-Renamed": mask_key(f"Bearer {CANARY}")})
    with pytest.raises(MaskedSecretUnmatched) as raised:
        merge_tool_headers(incoming, stored)
    assert raised.value.where == "tools.definition:config.headers.X-Renamed"


async def test_the_tool_update_route_keeps_the_stored_header(monkeypatch):
    import api.routes.tool as tool_route

    definition = _tool_definition(
        {"Authorization": f"Bearer {CANARY}", "X-Tenant": "tenant-42"}
    )
    stored = MagicMock()
    stored.definition = copy.deepcopy(definition)
    saved = MagicMock()
    saved.definition = copy.deepcopy(definition)
    update = AsyncMock(return_value=saved)
    monkeypatch.setattr(
        tool_route.db_client, "get_tool_by_uuid", AsyncMock(return_value=stored)
    )
    monkeypatch.setattr(tool_route.db_client, "update_tool", update)
    monkeypatch.setattr(tool_route, "build_tool_response", lambda tool, **_: tool)
    user = MagicMock(selected_organization_id=1, id=1)

    await call_update_tool_route(
        "tu-1",
        request=UpdateToolRequest(definition=mask_tool_definition(definition)),
        user=user,
    )

    written = update.await_args.kwargs["definition"]
    assert written["config"]["headers"] == definition["config"]["headers"]


# --- run detail: references are shown as they are, never a value ---------------------------------------------------


def test_masking_leaves_references_and_redactions_readable():
    document = {
        "voicemail_detection": {"api_key": credential_ref(12)},
        "model_overrides": {"llm": {"api_key": REDACTED}, "tts": {"api_key": CANARY}},
    }
    masked = mask_secrets(document)
    assert masked["voicemail_detection"]["api_key"] == credential_ref(12)
    assert masked["model_overrides"]["llm"]["api_key"] == REDACTED
    assert masked["model_overrides"]["tts"]["api_key"] == mask_key(CANARY)


# --- VOZ-AC-B6-19: a BYOK credential is not returned in clear by the create/update/GET response -------------------


async def test_the_model_configuration_response_never_carries_the_key(monkeypatch):
    import api.routes.organization as organization_route
    from api.schemas.ai_model_configuration import OrganizationAIModelConfigurationV2
    from api.services.configuration.ai_model_configuration import (
        ResolvedAIModelConfiguration,
        compile_ai_model_configuration_v2,
    )

    configuration = OrganizationAIModelConfigurationV2.model_validate(
        {"version": 2, "mode": "dograh", "dograh": {"api_key": f"{CANARY}-byok"}}
    )
    resolved = ResolvedAIModelConfiguration(
        effective=compile_ai_model_configuration_v2(configuration),
        source="organization_v2",
        organization_configuration=configuration,
    )
    monkeypatch.setattr(
        organization_route,
        "get_resolved_ai_model_configuration",
        AsyncMock(return_value=resolved),
    )
    user = MagicMock(selected_organization_id=1)

    for explicit in (configuration, None):  # the PUT/POST echo and the GET
        response = await organization_route._model_configuration_v2_response(
            user=user, configuration=explicit
        )
        body = response.model_dump_json()
        assert CANARY not in body
        assert "****byok" in body


# --- the workflow routes, end to end (database) -----------------------------------------------------------------


@pytest.fixture
async def org_user(async_session):
    from api.db.models import OrganizationModel, UserModel

    org = OrganizationModel(provider_id="test-org-header-masking")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(
        provider_id="test-user-header-masking", selected_organization_id=org.id
    )
    async_session.add(user)
    await async_session.flush()
    return org, user


def _graph_with_hook(headers):
    return _graph(
        {"id": "1", "type": "startCall", "data": {"name": "Start", "prompt": "Hi"}},
        {"id": "2", "type": "endCall", "data": {"name": "End", "prompt": "Bye"}},
        _webhook(headers=headers),
    ) | {
        "edges": [
            {
                "id": "e",
                "source": "1",
                "target": "2",
                "data": {"label": "end", "condition": "x"},
            }
        ]
    }


async def test_the_workflow_routes_mask_headers_and_refuse_an_orphan_mask(
    test_client_factory, db_session, org_user
):
    org, user = org_user
    stored_headers = copy.deepcopy(STORED["nodes"][0]["data"]["custom_headers"])
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition=_graph_with_hook(stored_headers),
        user_id=user.id,
        organization_id=org.id,
    )
    async with test_client_factory(user) as client:
        read = await client.get(f"/api/v1/workflow/fetch/{workflow.id}")
        assert read.status_code == 200 and CANARY not in read.text
        masked_graph = read.json()["workflow_definition"]

        kept = await client.put(
            f"/api/v1/workflow/{workflow.id}",
            json={"workflow_definition": masked_graph},
        )
        assert kept.status_code == 200, kept.text
        assert CANARY not in kept.text

        orphan = copy.deepcopy(masked_graph)
        hook = next(node for node in orphan["nodes"] if node["id"] == "hook")
        hook["data"]["custom_headers"][0]["key"] = "X-Renamed"
        refused = await client.put(
            f"/api/v1/workflow/{workflow.id}", json={"workflow_definition": orphan}
        )
    assert refused.status_code == 422, refused.text
    error = refused.json()["error"]
    assert error["code"] == "masked_secret_unmatched"
    assert (
        error["where"]
        == "workflow_definition:nodes[hook].data.custom_headers[X-Renamed]"
    )
    assert error["reason"] and error["hint"]

    draft = await db_session.get_draft_version(workflow.id)
    hook = next(node for node in draft.workflow_json["nodes"] if node["id"] == "hook")
    assert hook["data"]["custom_headers"] == stored_headers
