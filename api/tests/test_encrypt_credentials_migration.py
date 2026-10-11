"""e2b7c4d9a613 on real SQL: encrypt every registered secret leaf in place, verify, and decrypt on downgrade.

Seeds every registered surface in every row shape (a full document, a double-encoded one, a document without
secrets, empty leaves and run markers, JSON null, and a row the app already encrypted), then runs upgrade, upgrade
again, downgrade and upgrade. The app (api/db/credentials_at_rest.py) must read what the migration wrote and the
migration must decrypt what the app wrote (VOZ-N0-24c, VOZ-AC-B9-38).
"""

import base64
import json
import logging
import os

import pytest
from sqlalchemy import text

from api.services.security import credential_box
from api.services.security.secret_surfaces import SURFACES
from api.tests._at_rest_seed import CANARY, TOKEN, build_row, documents_for, new_context
from api.tests._migration_sql import load_migration, run_downgrade, run_upgrade

REVISION = "e2b7c4d9a613"
KEY = base64.b64encode(os.urandom(32)).decode()
# Seeded from the surfaces frozen in the migration, not the registry at HEAD: a surface registered later is
# encrypted by its own migration, and must not make this one look broken.
FROZEN = load_migration(REVISION)._SURFACES
DOCUMENTS = documents_for(FROZEN)


def _emptied(document):
    """The document with every leaf replaced by an empty value or a run marker (nothing to encrypt)."""
    if isinstance(document, dict):
        return {
            key: {} if key == "credentials" else _emptied(value)
            for key, value in document.items()
        }
    if isinstance(document, list):
        return [_emptied(value) for value in document]
    return "<redacted>" if len(str(document)) % 2 else ""


SHAPES = {
    "full": lambda doc, whole: doc,
    "double_encoded": lambda doc, whole: json.dumps(doc),
    "without_secrets": lambda doc, whole: {"legacy": True},
    "empty_leaves": lambda doc, whole: {} if whole else _emptied(doc),
    "json_null": lambda doc, whole: None,
}
WHOLE = {
    (table, column, row_key) for table, column, path, row_key in FROZEN if not path
}


@pytest.fixture
def migration():
    """The migration, with the open-sessions check off: the test process holds other connections to the test DB.
    test_the_migration_refuses_while_other_sessions_are_connected runs the real check."""
    module = load_migration(REVISION)
    module._other_sessions = lambda conn: 0
    return module


def _set_key(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(credential_box.ENV, raising=False)
    else:
        monkeypatch.setenv(credential_box.ENV, value)
    credential_box.load_keys.cache_clear()


@pytest.fixture(autouse=True)
def _clear_key_cache():
    yield
    credential_box.load_keys.cache_clear()


async def _seed(async_session, monkeypatch):
    """``[(table, column, id, document)]``: every group in every shape written keyless (plaintext, as before the
    migration), plus every full document written by the app with the key (already tokens)."""
    rows = []
    for key_value, shapes in (
        (None, SHAPES),
        (f"k1={KEY}", {"app_encrypted": SHAPES["full"]}),
    ):
        _set_key(monkeypatch, key_value)
        for name, shape in shapes.items():
            ctx = await new_context(
                async_session
            )  # keyed stores allow one row per key per owner
            for group, document in DOCUMENTS.items():
                table, column, row_key = group
                value = shape(json.loads(json.dumps(document)), group in WHOLE)
                row = build_row(ctx, table, column, value, row_key)
                async_session.add(row)
                await async_session.flush()
                rows.append(
                    (
                        table,
                        column,
                        row.id,
                        value,
                        name + ("/whole" if group in WHOLE else ""),
                    )
                )
    return rows


async def _raw(async_session, table, column, row_id):
    return (
        await async_session.execute(
            text(f"SELECT {column}::text FROM {table} WHERE id = :id"), {"id": row_id}
        )
    ).scalar_one()


async def _raws(async_session, rows):
    return [
        await _raw(async_session, table, column, row_id)
        for table, column, row_id, _v, _n in rows
    ]


def _same(stored, expected):
    """Equal as documents; a double-encoded one compares decoded (the migration re-encodes it)."""
    if isinstance(stored, str) and isinstance(expected, str):
        return json.loads(stored) == json.loads(expected)
    return stored == expected


async def _app_reads(async_session, rows):
    from api.db.credentials_at_rest import unseal
    from api.db.models import Base

    columns = {
        (mapper.local_table.name, column.name): getattr(
            mapper.class_, mapper.get_property_by_column(column).key
        )
        for mapper in Base.registry.mappers
        for column in mapper.local_table.columns
    }
    for table, column, row_id, value, name in rows:
        raw = await _raw(async_session, table, column, row_id)
        stored = unseal(
            columns[(table, column)],
            row_id,
            json.loads(raw) if raw is not None else None,
        )
        assert _same(stored, value), (table, column, name, stored)


async def _canaries_anywhere(async_session):
    tables = (
        (
            await async_session.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
        )
        .scalars()
        .all()
    )
    found = []
    for table in tables:
        hits = (
            await async_session.execute(
                text(
                    f'SELECT count(*) FROM "{table}" t WHERE row_to_json(t)::text LIKE :c'
                ),
                {"c": f"%{CANARY}%"},
            )
        ).scalar_one()
        if hits:
            found.append((table, hits))
    return found


def test_every_frozen_surface_is_still_registered(migration):
    # The registry may grow (each new surface brings its own migration); it must not lose or change what this
    # migration encrypted, or the app would stop reading those tokens.
    registered = {(s.table, s.column, s.json_path, s.row_key) for s in SURFACES}
    assert set(migration._SURFACES) <= registered
    assert len(migration._SURFACES) == 108  # the registry at this revision


@pytest.mark.asyncio
async def test_every_row_shape_through_upgrade_and_downgrade(
    migration, async_session, monkeypatch
):
    rows = await _seed(async_session, monkeypatch)
    assert await _canaries_anywhere(async_session), "the keyless seed must be in clear"
    _set_key(monkeypatch, f"k1={KEY}")

    seeded = await _raws(async_session, rows)
    await run_upgrade(async_session, migration)
    assert await _canaries_anywhere(async_session) == []
    upgraded = await _raws(async_session, rows)
    # Nothing to encrypt in empty leaves, run markers, JSON null or tokens; a whole column is secret entirely.
    expected = {
        "full",
        "full/whole",
        "double_encoded",
        "double_encoded/whole",
        "without_secrets/whole",
    }
    for row, before, after in zip(rows, seeded, upgraded):
        assert (before != after) == (row[4] in expected), row[:3] + row[4:]
    assert sum(len(TOKEN.findall(raw or "")) for raw in upgraded) > len(DOCUMENTS) * 2
    await _app_reads(async_session, rows)
    table, column, row_id, value, _name = next(
        r for r in rows if r[0] == "tools" and r[4] == "full"
    )
    token = json.loads(upgraded[rows.index((table, column, row_id, value, _name))])[
        "config"
    ]["headers"]["k"]
    assert credential_box.decrypt(
        token, credential_box.aad_for(table, row_id, column, ("config", "headers", "k"))
    )

    await run_upgrade(async_session, migration)
    assert await _raws(async_session, rows) == upgraded, (
        "a second upgrade must change nothing"
    )

    await run_downgrade(async_session, migration)
    for (table, column, row_id, value, name), raw in zip(
        rows, await _raws(async_session, rows)
    ):
        assert not TOKEN.search(raw or ""), (table, column, name)
        assert _same(json.loads(raw) if raw is not None else None, value), (
            table,
            column,
            name,
        )

    await run_upgrade(async_session, migration)
    assert await _canaries_anywhere(async_session) == []
    await _app_reads(async_session, rows)


@pytest.mark.asyncio
async def test_without_a_key_outside_a_cell_the_upgrade_changes_nothing(
    migration, async_session, monkeypatch, caplog
):
    rows = await _seed(async_session, monkeypatch)
    _set_key(monkeypatch, None)
    monkeypatch.delenv("CELL_ROLE", raising=False)
    before = await _raws(async_session, rows)
    with caplog.at_level(logging.WARNING, logger="alembic.runtime.migration"):
        await run_upgrade(async_session, migration)
    assert await _raws(async_session, rows) == before
    (message,) = [
        r.getMessage() for r in caplog.records if "stay in clear" in r.getMessage()
    ]
    assert "code=credentials_at_rest_skipped" in message and "reason=" in message
    assert (
        "hint=" in message
        and "alembic downgrade c5e8a2d7f914 && alembic upgrade head" in message
    )


@pytest.mark.asyncio
async def test_without_a_key_in_a_cell_the_upgrade_refuses(
    migration, async_session, monkeypatch
):
    _set_key(monkeypatch, None)
    monkeypatch.setenv("CELL_ROLE", "api")
    with pytest.raises(
        RuntimeError, match="code=credentials_master_key_invalid"
    ) as caught:
        await run_upgrade(async_session, migration)
    assert "reason=" in str(caught.value) and "hint=" in str(caught.value)


@pytest.mark.asyncio
async def test_the_downgrade_refuses_without_the_key_only_when_tokens_exist(
    migration, async_session, monkeypatch
):
    _set_key(monkeypatch, None)
    await run_downgrade(
        async_session, migration
    )  # nothing encrypted: nothing to refuse

    _set_key(monkeypatch, f"k1={KEY}")
    ctx = await new_context(async_session)
    async_session.add(
        build_row(ctx, "integrations", "connection_details", {"token": CANARY}, None)
    )
    await async_session.flush()
    _set_key(monkeypatch, None)
    with pytest.raises(RuntimeError, match="code=credentials_master_key_invalid"):
        await run_downgrade(async_session, migration)


@pytest.mark.asyncio
async def test_a_row_left_in_clear_aborts_the_upgrade(
    migration, async_session, monkeypatch
):
    _set_key(monkeypatch, None)
    ctx = await new_context(async_session)
    async_session.add(
        build_row(ctx, "integrations", "connection_details", {"token": CANARY}, None)
    )
    await async_session.flush()
    _set_key(monkeypatch, f"k1={KEY}")
    monkeypatch.setattr(
        migration, "_write", lambda *args: None
    )  # the row is never rewritten
    with pytest.raises(
        RuntimeError, match="code=credentials_at_rest_verify_failed"
    ) as caught:
        await run_upgrade(async_session, migration)
    assert "where=integrations.connection_details" in str(caught.value)


@pytest.mark.asyncio
async def test_a_token_that_does_not_give_back_its_value_aborts_the_upgrade(
    migration, async_session, monkeypatch
):
    _set_key(monkeypatch, None)
    ctx = await new_context(async_session)
    row = build_row(ctx, "integrations", "connection_details", {"token": CANARY}, None)
    async_session.add(row)
    await async_session.flush()
    _set_key(monkeypatch, f"k1={KEY}")
    real = migration._encrypt
    monkeypatch.setattr(
        migration, "_encrypt", lambda keys, value, aad: real(keys, {"other": 1}, aad)
    )
    with pytest.raises(
        RuntimeError, match="code=credentials_at_rest_verify_failed"
    ) as caught:
        await run_upgrade(async_session, migration)
    assert f"integrations:{row.id}:connection_details:[]" in str(caught.value)
    assert CANARY not in str(caught.value)


@pytest.mark.asyncio
async def test_a_token_that_does_not_decrypt_aborts_the_upgrade(
    migration, async_session, monkeypatch
):
    _set_key(monkeypatch, f"k1={KEY}")
    ctx = await new_context(async_session)
    first, second = (
        build_row(ctx, "integrations", "connection_details", {"token": CANARY}, None)
        for _ in range(2)
    )
    async_session.add_all([first, second])
    await async_session.flush()
    await async_session.execute(
        text(
            "UPDATE integrations SET connection_details = "
            "(SELECT connection_details FROM integrations WHERE id = :a) WHERE id = :b"
        ),
        {"a": first.id, "b": second.id},
    )
    with pytest.raises(RuntimeError, match="code=credential_unreadable") as caught:
        await run_upgrade(async_session, migration)
    assert f"integrations:{second.id}:connection_details:[]" in str(caught.value)
    assert CANARY not in str(caught.value)


@pytest.mark.asyncio
async def test_the_migration_refuses_while_other_sessions_are_connected(
    async_session, test_engine, monkeypatch
):
    real = load_migration(REVISION)  # the real pg_stat_activity check
    _set_key(monkeypatch, f"k1={KEY}")
    async with test_engine.connect() as other:  # another process of some release
        await other.execute(text("SELECT 1"))
        for step in (run_upgrade, run_downgrade):
            with pytest.raises(
                RuntimeError, match="code=credentials_at_rest_sessions_open"
            ) as caught:
                await step(async_session, real)
            assert "reason=" in str(caught.value) and "hint=stop api, call, arq" in str(
                caught.value
            )


@pytest.mark.asyncio
async def test_a_refused_write_on_downgrade_prints_no_credential(
    migration, async_session, monkeypatch
):
    _set_key(monkeypatch, f"k1={KEY}")
    ctx = await new_context(async_session)
    row = build_row(ctx, "integrations", "connection_details", {"token": CANARY}, None)
    async_session.add(row)
    await async_session.flush()
    # A constraint the decrypted document breaks: the driver error would print it as a bound parameter.
    await async_session.execute(
        text(
            "ALTER TABLE integrations ADD CONSTRAINT no_canary_for_the_test "
            f"CHECK (connection_details::text NOT LIKE '%{CANARY}%') NOT VALID"
        )
    )
    with pytest.raises(
        RuntimeError, match="code=credentials_at_rest_write_failed"
    ) as caught:
        await run_downgrade(async_session, migration)
    assert f"where=integrations:{row.id}:connection_details" in str(caught.value)
    assert CANARY not in str(caught.value) and caught.value.__cause__ is None


UNREGISTERED = {
    ("organization_configurations", "value", "ORGANIZATION_BOOTSTRAP"): {
        "status": "completed",
        "owner_token": "u-1",
    },
    ("organization_configurations", "value", "WORKFLOW_CONFIGURATION_DEFAULTS"): {
        "model_overrides": {"llm": {"api_key": "u-2"}}
    },
    ("user_configurations", "configuration", "ONBOARDING"): {
        "api_key": "u-3",
        "step": 2,
    },
}


@pytest.mark.asyncio
async def test_rows_of_unregistered_keys_stay_byte_identical(
    migration, async_session, monkeypatch
):
    _set_key(monkeypatch, None)
    ctx = await new_context(async_session)
    rows = []
    for (table, column, row_key), document in UNREGISTERED.items():
        row = build_row(ctx, table, column, document, row_key)
        async_session.add(row)
        await async_session.flush()
        rows.append((table, column, row.id, document, row_key))
    call_events = build_row(
        ctx,
        "organization_configurations",
        "value",
        {"enabled": True, "config": {"private_key": CANARY, "project_id": "p1"}},
        "CALL_EVENTS",
    )
    async_session.add(call_events)
    await async_session.flush()
    seeded = await _raws(async_session, rows)

    _set_key(monkeypatch, f"k1={KEY}")
    await run_upgrade(async_session, migration)
    assert await _raws(async_session, rows) == seeded
    stored = json.loads(
        await _raw(
            async_session, "organization_configurations", "value", call_events.id
        )
    )
    assert stored["enabled"] is True and stored["config"]["project_id"] == "p1"
    assert TOKEN.fullmatch(stored["config"]["private_key"])

    await run_downgrade(async_session, migration)
    assert await _raws(async_session, rows) == seeded
