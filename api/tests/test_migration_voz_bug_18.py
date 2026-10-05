"""VOZ-BUG-18 (G0 §3.2 S3): upstream's migration rewrites only workflows/definitions and writes "default",
while unknown values fall back to DEFAULT_TURN_START_STRATEGY = min_words. Migrated and unmigrated rows
must resolve to the SAME start strategy."""

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from api.services.workflow.voz_bug_18 import backfill_turn_start, resolve_turn_start
from api.tests._migration_sql import load_migration, run_upgrade


def test_migrated_and_unmigrated_rows_resolve_the_same():
    migrated = backfill_turn_start({"turn_start_strategy": "provisional_vad"})
    assert resolve_turn_start(migrated) == resolve_turn_start({}) == "default"


def test_backfill_is_idempotent():
    doc = backfill_turn_start({"turn_start_strategy": "provisional_vad"})
    assert backfill_turn_start(dict(doc)) == doc


def test_backfill_preserves_explicit_min_words():
    assert (
        backfill_turn_start({"turn_start_strategy": "min_words"})["turn_start_strategy"]
        == "min_words"
    )


def test_count_before_equals_after():  # VOZ-AC-B9-38
    docs = [
        {"turn_start_strategy": "provisional_vad"},
        {},
        {"turn_start_strategy": "min_words"},
    ]
    after = [backfill_turn_start(dict(d)) for d in docs]
    assert len(after) == len(docs) and "provisional_vad" not in str(after)


# ---- 5be1d27c9a43 on real SQL ---------------------------------------------------------------

_LEGACY = {"turn_start_strategy": "provisional_vad", "max_call_duration": 600}
_DEFAULTS_KEY = "WORKFLOW_CONFIGURATION_DEFAULTS"


@pytest.fixture
async def seeded(db_session, async_session):
    """One org with a defaults row, a workflow + its definition and a run, all ids returned."""
    from api.db.models import OrganizationModel, UserModel
    from api.enums import CallType, WorkflowRunMode

    org = OrganizationModel(provider_id="test-org-voz-bug-18")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(
        provider_id="test-user-voz-bug-18", selected_organization_id=org.id
    )
    async_session.add(user)
    await async_session.flush()
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition={"nodes": [], "edges": []},
        user_id=user.id,
        organization_id=org.id,
    )
    run = await db_session.create_workflow_run(
        name="run",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        call_type=CallType.INBOUND,
        organization_id=org.id,
    )
    await db_session.upsert_configuration(org.id, _DEFAULTS_KEY, {})
    return SimpleNamespace(org_id=org.id, workflow_id=workflow.id, run_id=run.id)


async def _set(session, table, column, where, doc):
    await session.execute(
        text(f"UPDATE {table} SET {column} = CAST(:doc AS json) WHERE {where}"),
        {"doc": json.dumps(doc)},
    )


async def _docs(session, table, column, where) -> list:
    rows = await session.execute(
        text(f"SELECT {column}::text FROM {table} WHERE {where}")
    )
    docs = [json.loads(raw) for (raw,) in rows]
    assert docs, f"seed missing: {table} WHERE {where}"
    return docs


def _decoded(doc):
    return json.loads(doc) if isinstance(doc, str) else doc


def _stores(ids):
    return [
        (
            "organization_configurations",
            "value",
            f"organization_id = {ids.org_id} AND key = '{_DEFAULTS_KEY}'",
        ),
        ("workflow_runs", "effective_configurations", f"id = {ids.run_id}"),
        ("workflows", "workflow_configurations", f"id = {ids.workflow_id}"),
        (
            "workflow_definitions",
            "workflow_configurations",
            f"workflow_id = {ids.workflow_id}",
        ),
    ]


async def test_double_encoded_legacy_documents_are_migrated(async_session, seeded):
    """Fable-3: a document stored as a JSON string hides the key from `->>`; it must still migrate."""
    for table, column, where in _stores(seeded):
        await _set(async_session, table, column, where, json.dumps(_LEGACY))

    await run_upgrade(async_session, load_migration("5be1d27c9a43"))

    for table, column, where in _stores(seeded):
        for doc in await _docs(async_session, table, column, where):
            decoded = _decoded(doc)
            assert decoded["turn_start_strategy"] == "default", (table, doc)
            assert decoded["max_call_duration"] == 600, (table, doc)
