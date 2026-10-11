"""c5e8a2d7f914 on real SQL: historic runs keep references, not credential values (VOZ-N0-24b, VOZ-AC-B9-38).

Seeds one run per row shape (no snapshot, legacy document without secrets, values the pinned definition holds, a
value it does not hold, a double-encoded document, a section without its secret key, an already-redacted leaf, a
run with no definition, a definition stored double-encoded), then runs upgrade / downgrade / upgrade / downgrade.
"""

import copy
import json

import pytest
from sqlalchemy import text

from api.db.models import OrganizationModel, UserModel
from api.enums import WorkflowRunMode
from api.services.configuration.cascade import run_configurations_for
from api.services.configuration.credential_refs import (
    REDACTED,
    credential_ref,
    to_credential_refs,
)
from api.services.security.secret_surfaces import WORKFLOW_CONFIGURATION_SECRET_PATHS
from api.tests._migration_sql import load_migration, run_downgrade, run_upgrade

REVISION = "c5e8a2d7f914"
CANARY = "canary-historic-run-0002"

SECRETS = {
    "model_overrides": {"llm": {"provider": "openai", "api_key": f"{CANARY}-llm"}},
    "voicemail_detection": {"enabled": True, "api_key": f"{CANARY}-vm"},
}


@pytest.fixture
def migration():
    return load_migration(REVISION)


@pytest.fixture
async def org_user(async_session):
    org = OrganizationModel(provider_id="test-org-credrefs-migration")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(
        provider_id="test-user-credrefs-migration", selected_organization_id=org.id
    )
    async_session.add(user)
    await async_session.flush()
    return org, user


async def _seed(
    db_session,
    async_session,
    org,
    user,
    *,
    definition,
    effective,
    encoding="object",
    definition_encoding="object",
):
    """A run whose pinned definition stores ``definition`` and whose frozen document is ``effective``, written with
    SQL as an older release would have. ``encoding``: object, string (double-encoded) or null (SQL NULL)."""
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition={"nodes": [], "edges": []},
        user_id=user.id,
        organization_id=org.id,
    )
    definition_id = (
        await async_session.execute(
            text("SELECT min(id) FROM workflow_definitions WHERE workflow_id = :id"),
            {"id": workflow.id},
        )
    ).scalar_one()
    run = await db_session.create_workflow_run(
        name="run",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        organization_id=org.id,
        definition_id=definition_id,
        effective_configurations=None,
    )
    for table, row_id, column, doc, how in (
        (
            "workflow_definitions",
            definition_id,
            "workflow_configurations",
            definition,
            definition_encoding,
        ),
        ("workflow_runs", run.id, "effective_configurations", effective, encoding),
    ):
        if how == "null":
            continue
        stored = json.dumps(doc)
        await async_session.execute(
            text(f"UPDATE {table} SET {column} = CAST(:doc AS json) WHERE id = :id"),
            {"doc": json.dumps(stored) if how == "string" else stored, "id": row_id},
        )
    return run.id, definition_id


async def _raw(async_session, run_id):
    return (
        await async_session.execute(
            text(
                "SELECT effective_configurations::text FROM workflow_runs WHERE id = :id"
            ),
            {"id": run_id},
        )
    ).scalar_one()


def _doc(raw):
    """The stored document, decoding a double-encoded one."""
    value = json.loads(raw) if raw is not None else None
    return json.loads(value) if isinstance(value, str) else value


async def test_every_row_shape_through_upgrade_and_downgrade(
    db_session, async_session, org_user, migration
):
    org, user = org_user
    rows = {}
    rows["no_snapshot"] = await _seed(
        db_session,
        async_session,
        org,
        user,
        definition=SECRETS,
        effective=None,
        encoding="null",
    )
    rows["legacy_no_secrets"] = await _seed(
        db_session,
        async_session,
        org,
        user,
        definition={},
        effective={"max_call_duration": 300},
    )
    rows["held"] = await _seed(
        db_session, async_session, org, user, definition=SECRETS, effective=SECRETS
    )
    other = copy.deepcopy(SECRETS)
    other["voicemail_detection"]["api_key"] = "rotated-since"
    rows["not_held"] = await _seed(
        db_session, async_session, org, user, definition=other, effective=SECRETS
    )
    rows["double_encoded"] = await _seed(
        db_session,
        async_session,
        org,
        user,
        definition=SECRETS,
        effective=SECRETS,
        encoding="string",
    )
    rows["missing_keys"] = await _seed(
        db_session,
        async_session,
        org,
        user,
        definition={},
        effective={
            "model_overrides": {"llm": {"provider": "openai"}},
            "voicemail_detection": {"enabled": False},
        },
    )
    rows["definition_double_encoded"] = await _seed(
        db_session,
        async_session,
        org,
        user,
        definition=SECRETS,
        effective=SECRETS,
        definition_encoding="string",
    )
    rows["no_definition"] = await _seed(
        db_session, async_session, org, user, definition=SECRETS, effective=SECRETS
    )
    await async_session.execute(
        text("UPDATE workflow_runs SET definition_id = NULL WHERE id = :id"),
        {"id": rows["no_definition"][0]},
    )
    rows["already_redacted"] = await _seed(
        db_session,
        async_session,
        org,
        user,
        definition=SECRETS,
        effective={"voicemail_detection": {"api_key": REDACTED}},
    )
    before = {
        name: await _raw(async_session, run_id) for name, (run_id, _) in rows.items()
    }

    await run_upgrade(async_session, migration)
    after = {
        name: await _raw(async_session, run_id) for name, (run_id, _) in rows.items()
    }

    everything = (
        (
            await async_session.execute(
                text("SELECT effective_configurations::text FROM workflow_runs")
            )
        )
        .scalars()
        .all()
    )
    assert not any(CANARY in (raw or "") for raw in everything), (
        "a credential value survived the upgrade"
    )
    for unchanged in (
        "no_snapshot",
        "legacy_no_secrets",
        "missing_keys",
        "already_redacted",
    ):
        assert after[unchanged] == before[unchanged], unchanged
    held = _doc(after["held"])
    assert held["model_overrides"]["llm"]["api_key"] == credential_ref(rows["held"][1])
    assert held["voicemail_detection"]["api_key"] == credential_ref(rows["held"][1])
    assert held["model_overrides"]["llm"]["provider"] == "openai"
    not_held = _doc(after["not_held"])
    assert not_held["voicemail_detection"]["api_key"] == REDACTED
    assert not_held["model_overrides"]["llm"]["api_key"] == credential_ref(
        rows["not_held"][1]
    )
    assert isinstance(json.loads(after["double_encoded"]), str), (
        "double encoding is kept"
    )
    assert _doc(after["double_encoded"])["voicemail_detection"][
        "api_key"
    ] == credential_ref(rows["double_encoded"][1])
    for redacted in ("definition_double_encoded", "no_definition"):
        assert _doc(after[redacted])["voicemail_detection"]["api_key"] == REDACTED, (
            redacted
        )

    # The app reads the migrated row back with the values the definition holds.
    loaded = await db_session.get_workflow_run(rows["held"][0], organization_id=org.id)
    assert run_configurations_for(loaded) == SECRETS

    # Idempotent: a second pass changes nothing.
    await run_upgrade(async_session, migration)
    assert {
        name: await _raw(async_session, run_id) for name, (run_id, _) in rows.items()
    } == after

    # Downgrade resolves references back; a redaction is one-way.
    await run_downgrade(async_session, migration)
    down = {
        name: await _raw(async_session, run_id) for name, (run_id, _) in rows.items()
    }
    assert _doc(down["held"]) == SECRETS
    assert _doc(down["double_encoded"]) == SECRETS and isinstance(
        json.loads(down["double_encoded"]), str
    )
    assert _doc(down["not_held"])["voicemail_detection"]["api_key"] == REDACTED
    assert (
        _doc(down["not_held"])["model_overrides"]["llm"]["api_key"] == f"{CANARY}-llm"
    )
    for name in (
        "no_snapshot",
        "legacy_no_secrets",
        "missing_keys",
        "already_redacted",
        "no_definition",
        "definition_double_encoded",
    ):
        assert down[name] == after[name], name

    # down/up/down again lands on the same rows.
    await run_upgrade(async_session, migration)
    assert {
        name: await _raw(async_session, run_id) for name, (run_id, _) in rows.items()
    } == after
    await run_downgrade(async_session, migration)
    assert {
        name: await _raw(async_session, run_id) for name, (run_id, _) in rows.items()
    } == down


async def test_upgrade_aborts_when_a_value_is_left(
    db_session, async_session, org_user, migration, monkeypatch
):
    org, user = org_user
    await _seed(
        db_session, async_session, org, user, definition=SECRETS, effective=SECRETS
    )
    monkeypatch.setattr(migration, "_to_refs", lambda doc, pinned, definition_id: 0)

    with pytest.raises(RuntimeError, match="run_credentials_left"):
        await run_upgrade(async_session, migration)


def test_the_frozen_secret_paths_are_the_registry_at_this_revision(migration):
    # A path added to the registry later needs its own migration for the runs frozen before it.
    assert migration._SECRET_PATHS == WORKFLOW_CONFIGURATION_SECRET_PATHS


@pytest.mark.parametrize(
    "pinned",
    [SECRETS, {}, {"voicemail_detection": {"api_key": f"{CANARY}-vm"}}, None],
    ids=["all_held", "none_held", "partly_held", "no_definition"],
)
def test_the_frozen_rewrite_matches_the_app(migration, pinned):
    document = {
        **copy.deepcopy(SECRETS),
        "model_configuration_v2_override": {"dograh": {"api_key": [f"{CANARY}-a", ""]}},
        "service_tuning": {"tts": {"_all": {"ctor": {"url": f"wss://{CANARY}"}}}},
    }
    frozen = copy.deepcopy(document)
    migration._to_refs(frozen, pinned or {}, 11)
    assert frozen == to_credential_refs(document, pinned, 11)
