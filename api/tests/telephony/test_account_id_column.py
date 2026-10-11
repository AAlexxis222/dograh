"""VOZ-N0-24a: inbound routing and the setup-warning counts read no JSON inside `credentials`.

`telephony_configurations.account_id` is the plaintext routing key. Every writer fills it
through the db client, per provider; the lookups and the duplicate-account guard filter on it.
"""

import json
from uuid import uuid4

import pytest
from sqlalchemy import text

from api.db.models import OrganizationModel, TelephonyPhoneNumberModel

# provider -> (credentials as the UI saves them, expected account_id). Secrets sit next to the
# account field on purpose: Telnyx and Exotel call a secret `api_key`, Vonage's `api_key` is
# the account.
ACCOUNT_CASES = {
    "twilio": ({"account_sid": "AC1", "auth_token": "tok"}, "AC1"),
    "vonage": ({"api_key": "vk1", "api_secret": "sec", "signature_secret": "s"}, "vk1"),
    "telnyx": ({"api_key": "KEYSECRET", "connection_id": "conn1"}, "conn1"),
    "exotel": ({"account_sid": "ex1", "api_key": "KEYSECRET"}, "ex1"),
    "plivo": ({"auth_id": "pl1", "auth_token": "tok"}, "pl1"),
    "vobiz": ({"auth_id": "vb1", "auth_token": "tok"}, "vb1"),
    "cloudonix": ({"domain_id": "acme.cx.net", "bearer_token": "tok"}, "acme.cx.net"),
    "ari": ({"app_name": "dograh"}, None),
}


async def _org(async_session) -> OrganizationModel:
    organization = OrganizationModel(provider_id=f"acct-id-{uuid4()}")
    async_session.add(organization)
    await async_session.flush()
    return organization


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(ACCOUNT_CASES))
async def test_create_fills_the_account_id_of_the_provider(
    provider, async_session, db_session
):
    organization = await _org(async_session)
    credentials, expected = ACCOUNT_CASES[provider]

    row = await db_session.create_telephony_configuration(
        organization.id, "cfg", provider, credentials
    )

    assert row.account_id == expected


@pytest.mark.asyncio
async def test_update_follows_the_credentials_and_ignores_a_rename(
    async_session, db_session
):
    organization = await _org(async_session)
    row = await db_session.create_telephony_configuration(
        organization.id, "cfg", "twilio", {"account_sid": "AC1", "auth_token": "t"}
    )

    renamed = await db_session.update_telephony_configuration(
        row.id, organization.id, name="renamed"
    )
    assert renamed.account_id == "AC1"

    moved = await db_session.update_telephony_configuration(
        row.id,
        organization.id,
        credentials={"account_sid": "AC2", "auth_token": "t"},
    )
    assert moved.account_id == "AC2"

    cleared = await db_session.update_telephony_configuration(
        row.id, organization.id, credentials={"auth_token": "t"}
    )
    assert cleared.account_id is None


async def _number(async_session, organization, row, address):
    phone = TelephonyPhoneNumberModel(
        organization_id=organization.id,
        telephony_configuration_id=row.id,
        address=address,
        address_normalized=address,
        address_type="phone",
        is_active=True,
    )
    async_session.add(phone)
    await async_session.flush()
    return phone


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [p for p in sorted(ACCOUNT_CASES) if p != "ari"])
async def test_inbound_route_matches_on_the_account_id_column(
    provider, async_session, db_session
):
    organization = await _org(async_session)
    credentials, account_id = ACCOUNT_CASES[provider]
    row = await db_session.create_telephony_configuration(
        organization.id, "cfg", provider, credentials
    )
    phone = await _number(async_session, organization, row, "+15551230001")

    match = await db_session.find_inbound_route_by_account(
        provider=provider, account_id=account_id, to_number="+15551230001"
    )
    assert match is not None
    assert (match[0].id, match[1].id) == (row.id, phone.id)

    assert (
        await db_session.find_inbound_route_by_account(
            provider=provider, account_id="other", to_number="+15551230001"
        )
        is None
    )


@pytest.mark.asyncio
async def test_a_telnyx_api_key_is_not_a_routing_key(async_session, db_session):
    organization = await _org(async_session)
    row = await db_session.create_telephony_configuration(
        organization.id,
        "cfg",
        "telnyx",
        {"api_key": "KEYSECRET", "connection_id": "conn1"},
    )
    await _number(async_session, organization, row, "+15551230002")

    assert (
        await db_session.find_inbound_route_by_account(
            provider="telnyx", account_id="KEYSECRET", to_number="+15551230002"
        )
        is None
    )


@pytest.mark.asyncio
async def test_conflicts_are_found_on_the_account_id_column(async_session, db_session):
    organization = await _org(async_session)
    row = await db_session.create_telephony_configuration(
        organization.id, "cfg", "twilio", {"account_sid": "AC9", "auth_token": "t"}
    )
    await _number(async_session, organization, row, "+15551230003")

    found = await db_session.find_inbound_routing_conflicts(
        provider="twilio", account_id="AC9", addresses_normalized=["+15551230003"]
    )
    assert [(c.id, p.address) for c, p in found] == [(row.id, "+15551230003")]

    assert (
        await db_session.find_inbound_routing_conflicts(
            provider="twilio",
            account_id="AC9",
            addresses_normalized=["+15551230003"],
            exclude_configuration_id=row.id,
        )
        == []
    )
    assert (
        await db_session.find_inbound_routing_conflicts(
            provider="twilio",
            account_id="AC-other",
            addresses_normalized=["+15551230003"],
        )
        == []
    )


@pytest.mark.asyncio
async def test_setup_warning_counts_count_missing_and_empty_fields(
    async_session, db_session
):
    organization = await _org(async_session)
    other = await _org(async_session)
    create = db_session.create_telephony_configuration

    await create(organization.id, "t-absent", "telnyx", {"connection_id": "c1"})
    await create(
        organization.id,
        "t-empty",
        "telnyx",
        {"connection_id": "c2", "webhook_public_key": ""},
    )
    await create(
        organization.id,
        "t-null",
        "telnyx",
        {"connection_id": "c3", "webhook_public_key": None},
    )
    await create(
        organization.id,
        "t-set",
        "telnyx",
        {"connection_id": "c4", "webhook_public_key": "pk"},
    )
    await create(other.id, "t-other-org", "telnyx", {"connection_id": "c5"})
    parked = await create(
        organization.id, "t-inactive", "telnyx", {"connection_id": "c6"}
    )
    await db_session.set_telephony_configuration_inactive(
        parked.id, organization.id, "parked"
    )
    await create(organization.id, "v-absent", "vonage", {"api_key": "v1"})
    await create(
        organization.id, "v-empty", "vonage", {"api_key": "v2", "signature_secret": ""}
    )
    await create(
        organization.id,
        "v-set",
        "vonage",
        {"api_key": "v3", "signature_secret": "s"},
    )
    await create(organization.id, "tw", "twilio", {"account_sid": "AC1"})

    assert (
        await db_session.count_telnyx_configs_missing_webhook_public_key(
            organization.id
        )
        == 4  # absent, empty, null and the parked (inactive) one: the old SQL counted it
    )
    assert (
        await db_session.count_vonage_configs_missing_signature_secret(organization.id)
        == 2
    )
    assert (
        await db_session.count_telnyx_configs_missing_webhook_public_key(other.id) == 1
    )


@pytest.mark.asyncio
async def test_a_double_encoded_credentials_document_counts_as_missing(
    async_session, db_session
):
    organization = await _org(async_session)
    await async_session.execute(
        text(
            "INSERT INTO telephony_configurations "
            "(organization_id, name, provider, credentials) "
            "VALUES (:org, 'legacy', 'telnyx', CAST(:doc AS json))"
        ),
        {
            "org": organization.id,
            "doc": json.dumps(json.dumps({"webhook_public_key": "pk"})),
        },
    )

    assert (
        await db_session.count_telnyx_configs_missing_webhook_public_key(
            organization.id
        )
        == 1
    )


# --- rows the previous release saved: account_id NULL with valid credentials ---------


async def _null_out(async_session, row) -> None:
    await async_session.execute(
        text("UPDATE telephony_configurations SET account_id = NULL WHERE id = :id"),
        {"id": row.id},
    )


async def _stored(async_session, row, column="account_id"):
    result = await async_session.execute(
        text(f"SELECT {column} FROM telephony_configurations WHERE id = :id"),
        {"id": row.id},
    )
    return result.scalar_one()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [p for p in sorted(ACCOUNT_CASES) if p != "ari"])
async def test_a_row_with_a_null_account_id_still_routes_and_is_healed(
    provider, async_session, db_session
):
    organization = await _org(async_session)
    credentials, account_id = ACCOUNT_CASES[provider]
    row = await db_session.create_telephony_configuration(
        organization.id, "cfg", provider, credentials
    )
    phone = await _number(async_session, organization, row, "+15551230010")
    await _null_out(async_session, row)

    match = await db_session.find_inbound_route_by_account(
        provider=provider, account_id=account_id, to_number="+15551230010"
    )

    assert match is not None
    assert (match[0].id, match[1].id) == (row.id, phone.id)
    assert match[0].account_id == account_id
    assert await _stored(async_session, row) == account_id
    assert (
        await db_session.find_inbound_route_by_account(
            provider=provider, account_id=account_id, to_number="+15551230010"
        )
        is not None
    )


@pytest.mark.asyncio
async def test_the_fallback_skips_other_accounts_ari_inactive_and_other_orgs(
    async_session, db_session
):
    organization = await _org(async_session)
    other = await _org(async_session)
    create = db_session.create_telephony_configuration

    wrong = await create(organization.id, "wrong", "twilio", {"account_sid": "AC-w"})
    await _number(async_session, organization, wrong, "+15551230011")
    ari = await create(organization.id, "ari", "ari", {"app_name": "dograh"})
    await _number(async_session, organization, ari, "+15551230012")
    parked = await create(organization.id, "parked", "twilio", {"account_sid": "AC-p"})
    await _number(async_session, organization, parked, "+15551230013")
    await db_session.set_telephony_configuration_inactive(
        parked.id, organization.id, "parked"
    )
    foreign = await create(other.id, "foreign", "twilio", {"account_sid": "AC-f"})
    await _number(async_session, other, foreign, "+15551230014")
    for row in (wrong, ari, parked, foreign):
        await _null_out(async_session, row)

    async def route(account_id, number, **kwargs):
        return await db_session.find_inbound_route_by_account(
            provider="twilio", account_id=account_id, to_number=number, **kwargs
        )

    assert await route("AC-other", "+15551230011") is None
    assert await route("anything", "+15551230012") is None
    assert await route("AC-p", "+15551230013") is None
    assert await route("AC-f", "+15551230014", organization_id=organization.id) is None
    assert await route("AC-f", "+15551230014", organization_id=other.id) is not None
    for row in (wrong, ari, parked):
        assert await _stored(async_session, row) is None


@pytest.mark.asyncio
async def test_the_conflict_check_sees_a_row_with_a_null_account_id(
    async_session, db_session
):
    organization = await _org(async_session)
    row = await db_session.create_telephony_configuration(
        organization.id, "cfg", "twilio", {"account_sid": "AC7", "auth_token": "t"}
    )
    await _number(async_session, organization, row, "+15551230015")
    await _null_out(async_session, row)

    found = await db_session.find_inbound_routing_conflicts(
        provider="twilio", account_id="AC7", addresses_normalized=["+15551230015"]
    )

    assert [(c.id, p.address) for c, p in found] == [(row.id, "+15551230015")]
    assert await _stored(async_session, row) == "AC7"
    assert (
        await db_session.find_inbound_routing_conflicts(
            provider="twilio",
            account_id="AC7",
            addresses_normalized=["+15551230015"],
            exclude_configuration_id=row.id,
        )
        == []
    )


@pytest.mark.asyncio
async def test_healing_leaves_updated_at_alone(async_session, db_session):
    organization = await _org(async_session)
    row = await db_session.create_telephony_configuration(
        organization.id, "cfg", "twilio", {"account_sid": "AC8"}
    )
    await _number(async_session, organization, row, "+15551230016")
    await _null_out(async_session, row)
    before = await _stored(async_session, row, "updated_at")

    await db_session.find_inbound_route_by_account(
        provider="twilio", account_id="AC8", to_number="+15551230016"
    )

    assert await _stored(async_session, row, "account_id") == "AC8"
    assert await _stored(async_session, row, "updated_at") == before


@pytest.mark.asyncio
async def test_a_json_number_account_id_routes_as_its_text(async_session, db_session):
    organization = await _org(async_session)
    row = await db_session.create_telephony_configuration(
        organization.id, "cfg", "twilio", {"account_sid": 12345}
    )
    await _number(async_session, organization, row, "+15551230017")

    assert row.account_id == "12345"
    assert (
        await db_session.find_inbound_route_by_account(
            provider="twilio", account_id="12345", to_number="+15551230017"
        )
        is not None
    )
