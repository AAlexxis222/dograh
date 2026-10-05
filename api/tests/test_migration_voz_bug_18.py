"""VOZ-BUG-18 (G0 §3.2 S3): upstream's migration rewrites only workflows/definitions and writes "default";
5be1d27c9a43 applies the same mapping to every store. Migrated and unmigrated rows must resolve to the
SAME start strategy."""

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from api.services.workflow.voz_bug_18 import backfill_turn_start, resolve_turn_start
from api.tests._migration_sql import load_migration, run_upgrade


def test_backfill_is_idempotent():
    doc = backfill_turn_start({"turn_start_strategy": "provisional_vad"})
    assert backfill_turn_start(dict(doc)) == doc


def test_backfill_preserves_explicit_min_words():
    assert (
        backfill_turn_start({"turn_start_strategy": "min_words"})["turn_start_strategy"]
        == "min_words"
    )


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


async def _row_counts(session) -> dict[str, int]:
    tables = (
        "organization_configurations",
        "workflow_runs",
        "workflows",
        "workflow_definitions",
    )
    return {
        t: (await session.execute(text(f"SELECT count(*) FROM {t}"))).scalar_one()
        for t in tables
    }


async def test_migrated_and_unmigrated_rows_resolve_the_same(async_session, seeded):
    """Rows the backfill rewrites and rows it leaves alone start turns the same way."""
    stores = _stores(seeded)
    unmigrated = [{}, {"turn_start_strategy": None}, {}, {}]
    for (table, column, where), doc in zip(stores, unmigrated):
        await _set(async_session, table, column, where, doc)
    await run_upgrade(async_session, load_migration("5be1d27c9a43"))
    # VOZ-AC-B2-30: the org defaults document is pinned, not left to the schema default.
    (org_doc,) = await _docs(async_session, *stores[0])
    assert org_doc["turn_start_strategy"] == "default"
    untouched = [
        resolve_turn_start(doc)
        for table, column, where in stores
        for doc in await _docs(async_session, table, column, where)
    ]

    for table, column, where in stores:
        await _set(async_session, table, column, where, _LEGACY)
    await run_upgrade(async_session, load_migration("5be1d27c9a43"))
    migrated = [
        resolve_turn_start(doc)
        for table, column, where in stores
        for doc in await _docs(async_session, table, column, where)
    ]

    assert set(untouched) == set(migrated) == {"default"}


async def test_count_before_equals_after(async_session, seeded):  # VOZ-AC-B9-38
    """Same row count per table, and no row left that the backfill still has to rewrite."""
    for table, column, where in _stores(seeded):
        await _set(async_session, table, column, where, _LEGACY)
    before = await _row_counts(async_session)

    await run_upgrade(async_session, load_migration("5be1d27c9a43"))

    assert await _row_counts(async_session) == before
    for table, column, where in _stores(seeded):
        for doc in await _docs(async_session, table, column, where):
            assert _decoded(doc)["turn_start_strategy"] != "provisional_vad", table


async def test_backfill_aborts_when_rows_still_need_the_rewrite(async_session, seeded):
    """The guard counts rows still needing the rewrite after the pass and aborts on any."""
    for table, column, where in _stores(seeded):
        await _set(async_session, table, column, where, _LEGACY)
    migration = load_migration("5be1d27c9a43")
    # A transform that leaves the value in place: every legacy row is still pending afterwards.
    migration._TARGETS = tuple(
        (table, column, where, lambda doc: doc, pending)
        for table, column, where, _transform, pending in migration._TARGETS
    )

    with pytest.raises(RuntimeError, match="still need the rewrite"):
        await run_upgrade(async_session, migration)
