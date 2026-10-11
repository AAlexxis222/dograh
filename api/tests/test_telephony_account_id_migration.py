"""VOZ-N0-24a migration: rehearsal of upgrade and downgrade on every credential row shape.

The test database is at head, so the test seeds rows with raw SQL (the shapes the ORM would
refuse: legacy, double-encoded, JSON null), downgrades the revision, then upgrades it again:
the real DDL and the real backfill run on the seeded rows, inside the test's transaction.
"""

import json
from uuid import uuid4

import pytest
from sqlalchemy import text

import api.services.telephony.providers  # noqa: F401  (registers every spec)
from api.db.models import OrganizationModel
from api.services.telephony import registry
from api.tests._migration_sql import load_migration, run_downgrade, run_upgrade

REVISION = "a4f1c8e27b35"

# name -> (provider, the JSON text stored in `credentials`, expected account_id)
ROWS = {
    "twilio": ("twilio", json.dumps({"account_sid": "AC1", "auth_token": "t"}), "AC1"),
    "vonage-api-key-is-the-account": (
        "vonage",
        json.dumps({"api_key": "vk1", "api_secret": "s"}),
        "vk1",
    ),
    "telnyx-api-key-is-a-secret": (
        "telnyx",
        json.dumps({"api_key": "KEYSECRET", "connection_id": "conn1"}),
        "conn1",
    ),
    "telnyx-without-connection": ("telnyx", json.dumps({"api_key": "K"}), None),
    "exotel": (
        "exotel",
        json.dumps({"account_sid": "ex1", "api_key": "KEYSECRET"}),
        "ex1",
    ),
    "plivo": ("plivo", json.dumps({"auth_id": "pl1"}), "pl1"),
    "vobiz": ("vobiz", json.dumps({"auth_id": "vb1"}), "vb1"),
    "cloudonix": ("cloudonix", json.dumps({"domain_id": "acme.cx.net"}), "acme.cx.net"),
    "ari": ("ari", json.dumps({"app_name": "dograh", "account_sid": "x"}), None),
    "unknown-provider": ("legacy-carrier", json.dumps({"account_sid": "AC2"}), None),
    "empty-object": ("twilio", "{}", None),
    "empty-string": ("twilio", json.dumps({"account_sid": ""}), None),
    "json-null-field": ("twilio", json.dumps({"account_sid": None}), None),
    "numeric-field": ("twilio", json.dumps({"account_sid": 12345}), None),
    "json-null-document": ("twilio", "null", None),
    "double-encoded": (
        "twilio",
        json.dumps(json.dumps({"account_sid": "AC-double"})),
        None,
    ),
}


def test_the_frozen_field_map_is_the_registry_at_this_revision():
    registry_map = {
        spec.name: spec.account_id_credential_field
        for spec in registry.all_specs()
        if spec.account_id_credential_field
    }
    assert load_migration(REVISION)._ACCOUNT_ID_FIELDS == registry_map


async def _seed(async_session) -> dict[str, int]:
    organization = OrganizationModel(provider_id=f"acct-id-mig-{uuid4()}")
    async_session.add(organization)
    await async_session.flush()
    ids = {}
    for name, (provider, document, _) in ROWS.items():
        result = await async_session.execute(
            text(
                "INSERT INTO telephony_configurations "
                "(organization_id, name, provider, credentials) "
                "VALUES (:org, :name, :provider, CAST(:doc AS json)) RETURNING id"
            ),
            {
                "org": organization.id,
                "name": name,
                "provider": provider,
                "doc": document,
            },
        )
        ids[name] = result.scalar_one()
    return ids


async def _column_and_index(async_session) -> tuple[bool, bool]:
    column = await async_session.execute(
        text(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'telephony_configurations' AND column_name = 'account_id'"
        )
    )
    index = await async_session.execute(
        text(
            "SELECT count(*) FROM pg_indexes WHERE tablename = 'telephony_configurations' "
            "AND indexname = :name"
        ),
        {"name": load_migration(REVISION).INDEX_NAME},
    )
    return bool(column.scalar_one()), bool(index.scalar_one())


async def _account_ids(async_session, ids: dict[str, int]) -> dict[str, str | None]:
    result = await async_session.execute(
        text(
            "SELECT id, account_id FROM telephony_configurations WHERE id = ANY(:ids)"
        ),
        {"ids": list(ids.values())},
    )
    by_id = dict(result.all())
    return {name: by_id[row_id] for name, row_id in ids.items()}


@pytest.mark.asyncio
async def test_downgrade_then_upgrade_rebuilds_account_id_for_every_row_shape(
    async_session,
):
    migration = load_migration(REVISION)
    ids = await _seed(async_session)

    await run_downgrade(async_session, migration)
    assert await _column_and_index(async_session) == (False, False)

    await run_upgrade(async_session, migration)
    assert await _column_and_index(async_session) == (True, True)
    assert await _account_ids(async_session, ids) == {
        name: expected for name, (_, _, expected) in ROWS.items()
    }

    # A second pass changes nothing (the backfill is idempotent).
    await run_upgrade_backfill_only(async_session, migration)
    assert await _account_ids(async_session, ids) == {
        name: expected for name, (_, _, expected) in ROWS.items()
    }


async def run_upgrade_backfill_only(async_session, migration) -> None:
    connection = await async_session.connection()
    await connection.run_sync(migration.backfill)


@pytest.mark.asyncio
async def test_backfill_corrects_a_stale_value(
    async_session,
):
    migration = load_migration(REVISION)
    ids = await _seed(async_session)
    await async_session.execute(
        text("UPDATE telephony_configurations SET account_id = 'stale' WHERE id = :id"),
        {"id": ids["twilio"]},
    )

    await run_upgrade_backfill_only(async_session, migration)
    assert (await _account_ids(async_session, ids))["twilio"] == "AC1"
