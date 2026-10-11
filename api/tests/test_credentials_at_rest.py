"""VOZ-N0-24c: every registered secret leaf is stored as a credential_box token and read back as its value.

Through the ORM seam (api/db/credentials_at_rest.py): a round trip per registered surface, the copy and version
paths (each row's token is bound to that row), the Core writes and column selects the events do not see, the keyed
stores scoped to their secret row keys, no key outside a cell, an unreadable token failing closed, and inbound
routing on encrypted telephony credentials.
"""

import base64
import json
import os

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.inspection import inspect

from api.db import models as m
from api.enums import UserConfigurationKey
from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.security import credential_box
from api.services.security.secret_surfaces import SURFACES
from api.tests._at_rest_seed import CANARY, DOCUMENTS, TOKEN, build_row, new_context

KEY = base64.b64encode(os.urandom(32)).decode()


@pytest.fixture
def master_key(monkeypatch):
    monkeypatch.setenv(credential_box.ENV, f"k1={KEY}")
    credential_box.load_keys.cache_clear()
    yield
    credential_box.load_keys.cache_clear()


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv(credential_box.ENV, raising=False)
    credential_box.load_keys.cache_clear()
    yield
    credential_box.load_keys.cache_clear()


@pytest.fixture
async def ctx(async_session):
    return await new_context(async_session)


async def _raw(session, table, column, row_id):
    return (
        await session.execute(
            text(f"SELECT {column}::text FROM {table} WHERE id = :id"), {"id": row_id}
        )
    ).scalar_one()


async def _reload(session, row):
    session.expunge_all()
    return (
        await session.execute(select(type(row)).where(type(row).id == row.id))
    ).scalar_one()


def test_every_registered_surface_is_seeded():
    seeded = {group for group in DOCUMENTS}
    assert seeded == {(s.table, s.column, s.row_key) for s in SURFACES}


@pytest.mark.asyncio
@pytest.mark.parametrize("group", sorted(DOCUMENTS, key=str), ids=str)
async def test_round_trip_through_the_orm(group, master_key, ctx, async_session):
    table, column, row_key = group
    document = DOCUMENTS[group]
    row = build_row(ctx, table, column, json.loads(json.dumps(document)), row_key)
    async_session.add(row)
    await async_session.flush()

    raw = await _raw(async_session, table, column, row.id)
    assert CANARY not in raw and TOKEN.search(raw), raw
    attribute = (
        inspect(type(row)).get_property_by_column(type(row).__table__.c[column]).key
    )
    assert getattr(row, attribute) == document
    assert not async_session.dirty

    loaded = await _reload(async_session, row)
    assert getattr(loaded, attribute) == document
    assert not inspect(loaded).modified

    async_session.expire(loaded)
    await async_session.refresh(loaded)
    assert getattr(loaded, attribute) == document
    assert not inspect(loaded).modified


@pytest.mark.asyncio
async def test_an_expired_row_decrypts_on_access_with_a_production_session(
    master_key, ctx, db_connection
):
    # The production sessionmaker expires on commit; the next attribute access is a refresh.
    maker = async_sessionmaker(bind=db_connection, expire_on_commit=True)
    async with maker() as session:
        row = m.ToolModel(
            organization_id=ctx["org"].id,
            name="t",
            created_by=ctx["user"].id,
            definition={"config": {"headers": {"X-Key": CANARY}}},
        )
        session.add(row)
        await session.flush()
        row_id = row.id
        await session.commit()  # expires the row; a query re-populates it (a refresh of the expired attributes)
        loaded = (
            await session.execute(select(m.ToolModel).where(m.ToolModel.id == row_id))
        ).scalar_one()
        assert loaded is row and row.definition == {
            "config": {"headers": {"X-Key": CANARY}}
        }
        assert not inspect(row).modified
        session.expire(row, ["definition"])
        await session.refresh(row, ["definition"])
        assert row.definition["config"]["headers"]["X-Key"] == CANARY


@pytest.mark.asyncio
async def test_an_insert_takes_its_id_before_the_row_is_written(
    master_key, ctx, async_session
):
    row = m.ExternalCredentialModel(
        organization_id=ctx["org"].id,
        name="c",
        created_by=ctx["user"].id,
        credential_data={"token": CANARY},
    )
    async_session.add(row)
    await async_session.flush()
    raw = await _raw(async_session, "external_credentials", "credential_data", row.id)
    token = json.loads(raw)
    assert credential_box.decrypt(
        token,
        credential_box.aad_for("external_credentials", row.id, "credential_data", ()),
    )


@pytest.mark.asyncio
async def test_every_surface_table_has_an_id_sequence(async_session):
    for table in sorted({s.table for s in SURFACES}):
        sequence = (
            await async_session.execute(
                text("SELECT pg_get_serial_sequence(:t, 'id')"), {"t": table}
            )
        ).scalar_one()
        assert sequence, table


@pytest.mark.asyncio
async def test_empty_values_and_run_markers_are_not_encrypted(
    master_key, ctx, async_session
):
    document = {
        "model_overrides": {"llm": {"api_key": "<redacted>"}, "tts": {"api_key": ""}},
        "voicemail_detection": {"api_key": None},
        "model_configuration_v2_override": {"x": {"credentials": {}}},
    }
    row = build_row(ctx, "workflow_runs", "effective_configurations", document, None)
    async_session.add(row)
    await async_session.flush()
    raw = await _raw(async_session, "workflow_runs", "effective_configurations", row.id)
    assert not TOKEN.search(raw) and json.loads(raw) == document


@pytest.mark.asyncio
async def test_a_secret_object_holding_another_secret_is_encrypted_once_as_a_whole(
    master_key, ctx, async_session
):
    document = {
        "model_configuration_v2_override": {
            "llm": {"credentials": {"api_key": CANARY, "region": "eu"}}
        }
    }
    row = build_row(ctx, "workflows", "workflow_configurations", document, None)
    async_session.add(row)
    await async_session.flush()
    stored = json.loads(
        await _raw(async_session, "workflows", "workflow_configurations", row.id)
    )
    assert TOKEN.fullmatch(
        stored["model_configuration_v2_override"]["llm"]["credentials"]
    )
    assert (await _reload(async_session, row)).workflow_configurations == document


@pytest.mark.asyncio
async def test_an_update_that_does_not_touch_the_secret_column_leaves_its_token(
    master_key, ctx, db_session, async_session
):
    row = await db_session.create_telephony_configuration(
        ctx["org"].id, "cfg", "twilio", {"account_sid": "AC9", "auth_token": CANARY}
    )
    before = await _raw(
        async_session, "telephony_configurations", "credentials", row.id
    )
    await db_session.update_telephony_configuration(
        row.id, ctx["org"].id, name="renamed"
    )
    assert (
        await _raw(async_session, "telephony_configurations", "credentials", row.id)
        == before
    )


# --- copies and versions ------------------------------------------------------------------------------------------

GRAPH = {
    "nodes": [
        {"id": "qa", "data": {"qa_api_key": f"{CANARY}-qa"}},
        {
            "id": "hook",
            "data": {"custom_headers": [{"key": "X-A", "value": f"{CANARY}-h1"}]},
        },
    ]
}
CONFIG = {
    "model_overrides": {"llm": {"provider": "openai", "api_key": f"{CANARY}-llm"}}
}


async def _definitions(session, workflow_id):
    rows = await session.execute(
        text(
            "SELECT id, workflow_json::text, workflow_configurations::text FROM workflow_definitions "
            "WHERE workflow_id = :w ORDER BY id"
        ),
        {"w": workflow_id},
    )
    return rows.all()


@pytest.mark.asyncio
async def test_every_copy_and_version_is_encrypted_for_its_own_row(
    master_key, ctx, db_session, async_session
):
    org, user = ctx["org"], ctx["user"]
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition=GRAPH,
        user_id=user.id,
        organization_id=org.id,
        workflow_configurations=CONFIG,
    )
    await db_session.save_workflow_draft(
        workflow.id, workflow_definition=GRAPH
    )  # new draft copied from V1
    await db_session.publish_workflow_draft(
        workflow.id
    )  # the workflow row copies the draft
    first = (await _definitions(async_session, workflow.id))[0][0]
    await db_session.revert_to_version(workflow.id, first)  # a new draft copied from V1

    rows = await _definitions(async_session, workflow.id)
    workflow_raw = (
        await async_session.execute(
            text(
                "SELECT workflow_definition::text, workflow_configurations::text FROM workflows WHERE id = :id"
            ),
            {"id": workflow.id},
        )
    ).one()
    stored = [*(cell for row in rows for cell in row[1:]), *workflow_raw]
    assert len(rows) == 3 and all(CANARY not in cell for cell in stored), stored
    tokens = [token for cell in stored for token in TOKEN.findall(cell)]
    assert len(tokens) == len(set(tokens)) == 3 * 3 + 3, (
        "a token was copied instead of re-encrypted"
    )

    async_session.expunge_all()
    for definition_id, *_ in rows:
        definition = await async_session.get(m.WorkflowDefinitionModel, definition_id)
        assert (definition.workflow_json, definition.workflow_configurations) == (
            GRAPH,
            CONFIG,
        )
    reloaded = await async_session.get(m.WorkflowModel, workflow.id)
    assert (reloaded.workflow_definition, reloaded.workflow_configurations) == (
        GRAPH,
        CONFIG,
    )


@pytest.mark.asyncio
async def test_a_token_copied_verbatim_to_another_row_does_not_decrypt(
    master_key, ctx, db_session, async_session
):
    org, user = ctx["org"], ctx["user"]
    source = await db_session.create_workflow(
        name="A",
        workflow_definition={},
        user_id=user.id,
        organization_id=org.id,
        workflow_configurations=CONFIG,
    )
    target = await db_session.create_workflow(
        name="B",
        workflow_definition={},
        user_id=user.id,
        organization_id=org.id,
        workflow_configurations={},
    )
    await async_session.execute(
        text(
            "UPDATE workflows SET workflow_configurations = "
            "(SELECT workflow_configurations FROM workflows WHERE id = :s) WHERE id = :t"
        ),
        {"s": source.id, "t": target.id},
    )
    async_session.expunge_all()
    with pytest.raises(credential_box.CredentialUnreadable) as caught:
        await async_session.get(m.WorkflowModel, target.id)
    assert (
        caught.value.where
        == f'workflows:{target.id}:workflow_configurations:["model_overrides","llm","api_key"]'
    )
    assert CANARY not in str(caught.value)


@pytest.mark.asyncio
async def test_writing_another_rows_token_fails_at_flush(
    master_key, ctx, db_session, async_session
):
    org, user = ctx["org"], ctx["user"]
    source = await db_session.create_workflow(
        name="A",
        workflow_definition={},
        user_id=user.id,
        organization_id=org.id,
        workflow_configurations=CONFIG,
    )
    raw = json.loads(
        await _raw(async_session, "workflows", "workflow_configurations", source.id)
    )
    async_session.add(
        m.WorkflowModel(name="B", organization_id=org.id, workflow_configurations=raw)
    )
    with pytest.raises(credential_box.CredentialUnreadable):
        await async_session.flush()


# --- Core writes and column selects -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_core_updates_of_credentials_and_tools_are_encrypted(
    master_key, ctx, db_session, async_session
):
    org, user = ctx["org"], ctx["user"]
    credential = await db_session.create_credential(
        organization_id=org.id,
        user_id=user.id,
        name="c",
        credential_type="bearer_token",
        credential_data={},
    )
    await db_session.update_credential(
        credential.credential_uuid, org.id, credential_data={"token": CANARY}
    )
    tool = await db_session.create_tool(
        organization_id=org.id,
        user_id=user.id,
        name="t",
        definition={"config": {"headers": {}}},
    )
    headers = {"config": {"headers": {"X-Key": CANARY}}}
    await db_session.update_tool(tool.tool_uuid, org.id, definition=headers)

    for table, column, row_id in (
        ("external_credentials", "credential_data", credential.id),
        ("tools", "definition", tool.id),
    ):
        raw = await _raw(async_session, table, column, row_id)
        assert CANARY not in raw and TOKEN.search(raw), raw
    async_session.expunge_all()
    assert (
        await db_session.get_credential_by_uuid(credential.credential_uuid, org.id)
    ).credential_data == {"token": CANARY}
    assert (
        await db_session.get_tool_by_uuid(tool.tool_uuid, org.id)
    ).definition == headers


@pytest.mark.asyncio
async def test_core_updates_of_workflow_configurations_are_encrypted(
    master_key, ctx, db_session, async_session
):
    org, user = ctx["org"], ctx["user"]
    workflow = await db_session.create_workflow(
        name="W", workflow_definition={}, user_id=user.id, organization_id=org.id
    )
    draft = await db_session.save_workflow_draft(
        workflow.id, workflow_configurations={}
    )
    await db_session.update_definition_configurations(draft.id, CONFIG)
    published = (await _definitions(async_session, workflow.id))[0][0]
    await db_session.bulk_update_workflow_model_configurations(
        organization_id=org.id,
        workflow_updates=[(workflow.id, CONFIG)],
        definition_updates=[(published, CONFIG)],
    )

    for table, row_id in (
        ("workflow_definitions", draft.id),
        ("workflow_definitions", published),
        ("workflows", workflow.id),
    ):
        raw = await _raw(async_session, table, "workflow_configurations", row_id)
        assert CANARY not in raw and TOKEN.search(raw), (table, raw)


@pytest.mark.asyncio
async def test_column_selects_return_plaintext(
    master_key, ctx, db_session, async_session
):
    org, user = ctx["org"], ctx["user"]
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition={},
        user_id=user.id,
        organization_id=org.id,
        workflow_configurations=CONFIG,
    )
    draft = await db_session.save_workflow_draft(
        workflow.id, workflow_configurations=CONFIG
    )
    published = (await _definitions(async_session, workflow.id))[0][0]
    legacy_run = build_row(
        ctx, "workflow_runs", "effective_configurations", CONFIG, None
    )
    legacy_run.workflow_id, legacy_run.definition_id = workflow.id, published
    pinned_run = build_row(ctx, "workflow_runs", "effective_configurations", None, None)
    pinned_run.workflow_id, pinned_run.definition_id = workflow.id, published
    async_session.add_all([legacy_run, pinned_run])
    await async_session.flush()
    async_session.expunge_all()

    assert await db_session.get_definition_configurations_with_owner(published) == (
        CONFIG,
        org.id,
    )
    listed = await db_session.list_draft_definitions_for_backfill(
        after_id=draft.id - 1, limit=1
    )
    assert listed == [(draft.id, workflow.id, org.id, CONFIG)]
    assert await db_session.get_workflow_run_config_without_credentials(legacy_run.id, org.id) == CONFIG
    assert await db_session.get_workflow_run_config_without_credentials(pinned_run.id, org.id) == CONFIG


@pytest.mark.asyncio
async def test_user_configurations_encrypt_only_the_model_configuration(
    master_key, ctx, db_session, async_session
):
    user = ctx["user"]
    configuration = EffectiveAIModelConfiguration.model_validate(
        {"llm": {"provider": "openai", "api_key": CANARY, "model": "gpt-4.1"}}
    )
    await db_session.update_user_configuration(user.id, configuration)
    await db_session.upsert_user_configuration_value(
        user.id, UserConfigurationKey.ONBOARDING.value, {"step": CANARY}
    )

    rows = dict(
        (
            await async_session.execute(
                text(
                    "SELECT key, configuration::text FROM user_configurations WHERE user_id = :u"
                ),
                {"u": user.id},
            )
        ).all()
    )
    assert CANARY not in rows["MODEL_CONFIGURATION"] and TOKEN.search(
        rows["MODEL_CONFIGURATION"]
    )
    assert json.loads(rows["ONBOARDING"]) == {"step": CANARY}
    async_session.expunge_all()
    assert (await db_session.get_user_configurations(user.id)).llm.api_key == CANARY

    with pytest.raises(ValueError, match="user_configuration_secret_key"):
        await db_session.upsert_user_configuration_value(
            user.id,
            UserConfigurationKey.MODEL_CONFIGURATION.value,
            {"llm": {"api_key": CANARY}},
        )


@pytest.mark.asyncio
async def test_org_rows_of_other_keys_are_not_encrypted(
    master_key, ctx, db_session, async_session
):
    await db_session.upsert_configuration(
        ctx["org"].id, "ORGANIZATION_PREFERENCES", {"timezone": CANARY}
    )
    raw = (
        await async_session.execute(
            text(
                "SELECT value::text FROM organization_configurations WHERE organization_id = :o"
            ),
            {"o": ctx["org"].id},
        )
    ).scalar_one()
    assert json.loads(raw) == {"timezone": CANARY}


# --- no key, unreadable tokens, inbound routing -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_a_key_values_are_stored_in_clear_and_a_warning_says_so(
    no_key, ctx, async_session
):
    from loguru import logger

    from api.db.credentials_at_rest import warn_if_plaintext_at_rest

    row = build_row(
        ctx, "tools", "definition", {"config": {"headers": {"X-Key": CANARY}}}, None
    )
    async_session.add(row)
    await async_session.flush()
    assert CANARY in await _raw(async_session, "tools", "definition", row.id)

    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        warn_if_plaintext_at_rest()
    finally:
        logger.remove(sink)
    assert any(
        "code=credentials_at_rest_plaintext" in message for message in messages
    ), messages


@pytest.mark.asyncio
async def test_with_a_key_no_warning_is_logged(master_key):
    from loguru import logger

    from api.db.credentials_at_rest import warn_if_plaintext_at_rest

    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        warn_if_plaintext_at_rest()
    finally:
        logger.remove(sink)
    assert not messages


@pytest.mark.asyncio
async def test_a_token_without_the_key_fails_closed(
    monkeypatch, master_key, ctx, async_session
):
    row = build_row(ctx, "integrations", "connection_details", {"token": CANARY}, None)
    async_session.add(row)
    await async_session.flush()
    monkeypatch.delenv(credential_box.ENV)
    credential_box.load_keys.cache_clear()
    async_session.expunge_all()
    with pytest.raises(credential_box.CredentialUnreadable) as caught:
        await async_session.get(m.IntegrationModel, row.id)
    error = caught.value
    assert error.code == "credential_unreadable" and error.reason and error.hint
    assert error.where == f"integrations:{row.id}:connection_details:[]"


@pytest.mark.asyncio
async def test_the_api_answers_an_unreadable_credential_with_a_named_error():
    from api.app import handle_credential_unreadable

    error = credential_box.CredentialUnreadable(
        'tools:7:definition:["config"]', "authentication failed"
    )
    response = await handle_credential_unreadable(None, error)
    body = json.loads(response.body)
    assert response.status_code == 500
    assert body["error"]["code"] == "credential_unreadable"
    assert body["error"]["where"] == 'tools:7:definition:["config"]'
    assert body["error"]["reason"] and body["error"]["hint"]


@pytest.mark.asyncio
async def test_inbound_routing_works_on_encrypted_telephony_credentials(
    master_key, ctx, db_session, async_session
):
    org = ctx["org"]
    credentials = {"account_sid": "AC-at-rest", "auth_token": CANARY}
    row = await db_session.create_telephony_configuration(
        org.id, "cfg", "twilio", credentials
    )
    async_session.add(
        m.TelephonyPhoneNumberModel(
            organization_id=org.id,
            telephony_configuration_id=row.id,
            address="+15551239876",
            address_normalized="+15551239876",
            address_type="phone",
            is_active=True,
        )
    )
    await async_session.flush()
    raw = await _raw(async_session, "telephony_configurations", "credentials", row.id)
    assert CANARY not in raw and "AC-at-rest" not in raw

    # A row whose routing key was never filled is healed from its decrypted credentials.
    await async_session.execute(
        text("UPDATE telephony_configurations SET account_id = NULL WHERE id = :id"),
        {"id": row.id},
    )
    async_session.expunge_all()
    match = await db_session.find_inbound_route_by_account(
        provider="twilio", account_id="AC-at-rest", to_number="+15551239876"
    )
    assert match is not None and match[0].id == row.id
    assert match[0].credentials == credentials
    match = await db_session.find_inbound_route_by_account(
        provider="twilio", account_id="AC-at-rest", to_number="+15551239876"
    )
    assert match is not None and match[0].credentials == credentials
