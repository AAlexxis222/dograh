"""VOZ-N0-30: public download links expire, can be revoked for good, and are off inside a cell.

Real Postgres for the mint, revoke and gate rules (the run row is the source of truth), the real route through ASGI
for the 404 codes, and the real route functions and payload builders for "links off leaks nothing".
"""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.db import db_client
from api.db.models import (
    OrganizationModel,
    UserModel,
    WorkflowModel,
    WorkflowRunModel,
)
from api.routes import public_download
from api.routes.organization_usage import get_usage_history
from api.routes.workflow import get_workflow_run
from api.services import public_links
from api.services.integrations.noveum.completion import (
    _build_recording_url as noveum_recording_url,
)
from api.services.integrations.tuner.completion import (
    _build_recording_url as tuner_recording_url,
)
from api.services.reports.run_report import build_run_report_csv
from api.services.runtime.durations import (
    DEFAULT_PUBLIC_LINK_TTL_H,
    PUBLIC_LINK_TTL_ENV,
    Durations,
    DurationsError,
)
from api.services.workflow.dto import WebhookNodeData
from api.tasks.run_integrations import _build_render_context, _build_webhook_payload

T = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
CONFIG_NAME = "WORKFLOW_CONFIGURATION_DEFAULTS"
# Distinct canaries: any of them showing up in a response or a payload is a leak.
CANARY_BUCKET = "internal-bucket-canary"
RECORDING_PATH = f"{CANARY_BUCKET}/recordings/mixed.wav"
TRANSCRIPT_PATH = f"{CANARY_BUCKET}/transcripts/t.txt"
USER_PATH = f"{CANARY_BUCKET}/recordings/user.wav"
BOT_PATH = f"{CANARY_BUCKET}/recordings/bot.wav"
# A link id already stored on a run (minted while links were on) and one handed to a payload builder.
OLD_LINK_ID = str(uuid.UUID(int=0xC0FFEE))
NEW_LINK_ID = "public-link"
PUBLIC_URL_FIELDS = (
    "transcript_public_url",
    "recording_public_url",
    "user_recording_public_url",
    "bot_recording_public_url",
)


@pytest.fixture(scope="module")
async def sessions(setup_test_database):
    engine = create_async_engine(setup_test_database)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    old_engine, old_session = db_client.engine, db_client.async_session
    db_client.engine, db_client.async_session = engine, factory
    yield factory
    db_client.engine, db_client.async_session = old_engine, old_session
    await engine.dispose()


@pytest.fixture
async def world(sessions):
    async with sessions() as session:
        org = OrganizationModel(provider_id=f"links-{uuid.uuid4().hex}")
        session.add(org)
        await session.flush()
        user = UserModel(
            provider_id=f"links-{uuid.uuid4().hex}", selected_organization_id=org.id
        )
        session.add(user)
        await session.flush()
        workflow = WorkflowModel(name="links", user_id=user.id, organization_id=org.id)
        session.add(workflow)
        await session.commit()

    async def make_run(**columns) -> WorkflowRunModel:
        async with sessions() as session:
            run = WorkflowRunModel(
                name="links-run",
                workflow_id=workflow.id,
                mode="twilio",
                usage_info={"call_duration_seconds": 5},
                **columns,
            )
            session.add(run)
            await session.commit()
            return run

    async def reload(run_id: int) -> WorkflowRunModel:
        async with sessions() as session:
            return (
                await session.execute(
                    select(WorkflowRunModel).where(WorkflowRunModel.id == run_id)
                )
            ).scalar_one()

    yield SimpleNamespace(
        org=org, user=user, workflow=workflow, make_run=make_run, reload=reload
    )

    async with sessions() as session:
        await session.execute(
            delete(WorkflowRunModel).where(WorkflowRunModel.workflow_id == workflow.id)
        )
        await session.execute(
            delete(WorkflowModel).where(WorkflowModel.id == workflow.id)
        )
        await session.execute(delete(UserModel).where(UserModel.id == user.id))
        await session.execute(
            delete(OrganizationModel).where(OrganizationModel.id == org.id)
        )
        await session.commit()


@pytest.fixture
def clock(monkeypatch):
    """The one clock of link minting and checking, moved by the test."""
    state = {"now": T}
    monkeypatch.setattr(public_links, "now", lambda: state["now"])
    return state


@pytest.fixture
def in_cell(monkeypatch):
    monkeypatch.setenv("CELL_ROLE", "api")


@pytest.fixture
def http():
    app = FastAPI()
    app.include_router(public_download.router)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def storage():
    fake = SimpleNamespace(
        aget_signed_url=AsyncMock(return_value="https://storage.example/signed")
    )
    with patch.object(public_download, "get_storage_for_backend", lambda _b: fake):
        yield fake


async def open_link(http, link_id, artifact="recording"):
    return await http.get(f"/public/download/workflow/{link_id}/{artifact}")


async def set_org_links(world, value):
    await db_client.upsert_configuration(
        world.org.id, CONFIG_NAME, {"public_links_enabled": value}
    )


# ---------------------------------------------------------------------------
# TTL knob (durations.py is the one source) and the pure policy
# ---------------------------------------------------------------------------


def test_ttl_defaults_to_72_hours():
    assert Durations.from_env({}).public_link_ttl_h == DEFAULT_PUBLIC_LINK_TTL_H == 72


def test_ttl_is_read_from_the_environment():
    assert Durations.from_env({PUBLIC_LINK_TTL_ENV: "24"}).public_link_ttl_h == 24


@pytest.mark.parametrize("raw", ["0", "-5"])
def test_a_non_positive_ttl_fails_startup_with_a_named_error(raw):
    with pytest.raises(DurationsError) as e:
        Durations.from_env({PUBLIC_LINK_TTL_ENV: raw})
    assert e.value.code == "knob_out_of_range"
    assert PUBLIC_LINK_TTL_ENV in e.value.reason and e.value.hint


def test_a_ttl_that_is_not_a_number_fails_startup_with_a_named_error():
    with pytest.raises(DurationsError) as e:
        Durations.from_env({PUBLIC_LINK_TTL_ENV: "3d"})
    assert e.value.code == "knob_invalid" and PUBLIC_LINK_TTL_ENV in e.value.reason


def test_a_link_is_expired_exactly_at_its_deadline_and_a_null_deadline_is_expired():
    deadline = T + timedelta(hours=72)
    assert not public_links.is_expired(deadline, deadline - timedelta(seconds=1))
    assert public_links.is_expired(deadline, deadline)
    assert public_links.is_expired(None, T)


def test_links_are_on_outside_a_cell_whatever_the_organization_says(monkeypatch):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    assert public_links.links_enabled({}) is True
    assert public_links.links_enabled({"public_links_enabled": False}) is True


@pytest.mark.parametrize(
    "document, enabled",
    [
        ({}, False),
        (None, False),
        ({"public_links_enabled": True}, True),
        ({"public_links_enabled": False}, False),
        ({"public_links_enabled": "true"}, False),
        ({"public_links_enabled": 1}, False),
        ("garbage", False),
    ],
)
def test_links_are_off_in_a_cell_unless_the_organization_says_exactly_true(
    in_cell, document, enabled
):
    assert public_links.links_enabled(document) is enabled


def test_carrier_region_policy_is_not_a_cell_marker(monkeypatch):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    monkeypatch.setenv("CARRIER_REGION_POLICY", "eu")
    assert public_links.links_enabled({}) is True


# ---------------------------------------------------------------------------
# Minting: the TTL counts from the mint, not from the call
# ---------------------------------------------------------------------------


async def test_a_new_link_expires_72_hours_after_it_was_minted(world, clock):
    run = await world.make_run()
    link_id = await db_client.ensure_public_access_token(run.id)
    stored = await world.reload(run.id)
    assert stored.public_access_token == link_id
    assert stored.public_access_expires_at == T + timedelta(hours=72)
    assert stored.public_access_revoked_at is None


async def test_ensure_is_idempotent_and_keeps_the_first_deadline(world, clock):
    run = await world.make_run()
    first = await db_client.ensure_public_access_token(run.id)
    clock["now"] = T + timedelta(hours=10)
    assert await db_client.ensure_public_access_token(run.id) == first
    stored = await world.reload(run.id)
    assert stored.public_access_expires_at == T + timedelta(hours=72)


async def test_a_late_mint_lives_72_hours_from_the_mint_not_from_the_run(
    world, clock, http, storage
):
    """Run at T, minted at T+2h: alive at T+73h, expired at T+74h."""
    run = await world.make_run(recording_url=RECORDING_PATH)
    clock["now"] = T + timedelta(hours=2)
    link_id = await db_client.ensure_public_access_token(run.id)

    clock["now"] = T + timedelta(hours=73)
    alive = await open_link(http, link_id)
    clock["now"] = T + timedelta(hours=74) - timedelta(seconds=1)
    just_before = await open_link(http, link_id)
    clock["now"] = T + timedelta(hours=74)
    expired = await open_link(http, link_id)

    assert alive.status_code == 302 and just_before.status_code == 302
    assert expired.status_code == 404
    assert expired.json()["detail"]["code"] == "link_expired"


async def test_a_row_with_a_link_and_no_deadline_is_expired(
    world, clock, http, storage
):
    """Rows minted before the column existed: the link id stays, the link does not open (fail closed)."""
    await world.make_run(recording_url=RECORDING_PATH, public_access_token=OLD_LINK_ID)
    resp = await open_link(http, OLD_LINK_ID)
    assert resp.status_code == 404 and resp.json()["detail"]["code"] == "link_expired"
    storage.aget_signed_url.assert_not_awaited()


async def test_an_expired_link_answers_in_the_b0_28_shape(world, clock, http, storage):
    run = await world.make_run(recording_url=RECORDING_PATH)
    link_id = await db_client.ensure_public_access_token(run.id)
    clock["now"] = T + timedelta(hours=100)
    detail = (await open_link(http, link_id)).json()["detail"]
    assert detail["code"] == "link_expired"
    assert detail["where"] and detail["reason"] and detail["hint"]
    assert link_id not in str(detail)


async def test_a_live_link_redirects_to_the_signed_url(world, clock, http, storage):
    run = await world.make_run(recording_url=RECORDING_PATH)
    link_id = await db_client.ensure_public_access_token(run.id)
    resp = await open_link(http, link_id)
    assert resp.status_code == 302
    assert resp.headers["location"] == "https://storage.example/signed"


async def test_an_unknown_link_is_a_404(world, clock, http, storage):
    resp = await open_link(http, str(uuid.uuid4()))
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "link_invalid"


# ---------------------------------------------------------------------------
# Revocation: final
# ---------------------------------------------------------------------------


async def test_a_revoked_link_is_a_404_and_the_id_is_gone(world, clock, http, storage):
    run = await world.make_run(transcript_url=TRANSCRIPT_PATH)
    link_id = await db_client.ensure_public_access_token(run.id)
    assert await db_client.revoke_public_access(run.id) is True

    stored = await world.reload(run.id)
    assert stored.public_access_token is None
    assert stored.public_access_revoked_at == T
    resp = await open_link(http, link_id, "transcript")
    assert resp.status_code == 404
    storage.aget_signed_url.assert_not_awaited()


async def test_a_revoked_run_is_never_minted_again(world, clock):
    run = await world.make_run()
    await db_client.ensure_public_access_token(run.id)
    await db_client.revoke_public_access(run.id)

    assert await db_client.ensure_public_access_token(run.id) is None
    assert (await world.reload(run.id)).public_access_token is None


async def test_revoking_twice_keeps_the_first_revocation_time(world, clock):
    run = await world.make_run()
    await db_client.ensure_public_access_token(run.id)
    await db_client.revoke_public_access(run.id)
    clock["now"] = T + timedelta(hours=5)
    await db_client.revoke_public_access(run.id)
    assert (await world.reload(run.id)).public_access_revoked_at == T


async def test_revoking_a_missing_run_is_false(world):
    assert await db_client.revoke_public_access(2_000_000_000) is False


# ---------------------------------------------------------------------------
# Links off in a cell
# ---------------------------------------------------------------------------


async def test_ensure_mints_nothing_in_a_cell_by_default(world, clock, in_cell):
    run = await world.make_run()
    assert await db_client.ensure_public_access_token(run.id) is None
    stored = await world.reload(run.id)
    assert stored.public_access_token is None
    assert stored.public_access_expires_at is None


async def test_ensure_mints_in_a_cell_once_the_organization_turns_links_on(
    world, clock, in_cell
):
    await set_org_links(world, True)
    run = await world.make_run()
    assert await db_client.ensure_public_access_token(run.id)
    stored = await world.reload(run.id)
    assert stored.public_access_expires_at == T + timedelta(hours=72)


async def test_an_organization_that_says_false_has_no_links_in_a_cell(
    world, clock, in_cell
):
    await set_org_links(world, False)
    run = await world.make_run()
    assert await db_client.ensure_public_access_token(run.id) is None


async def test_ensure_does_not_hand_out_a_stored_link_when_links_are_off(
    world, clock, in_cell
):
    run = await world.make_run(
        public_access_token=OLD_LINK_ID, public_access_expires_at=T + timedelta(days=1)
    )
    assert await db_client.ensure_public_access_token(run.id) is None


async def test_links_off_in_a_cell_answers_404_public_links_disabled_to_a_live_link(
    world, clock, in_cell, http, storage
):
    await world.make_run(
        recording_url=RECORDING_PATH,
        public_access_token=OLD_LINK_ID,
        public_access_expires_at=T + timedelta(days=1),
    )
    resp = await open_link(http, OLD_LINK_ID)
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "public_links_disabled"
    assert detail["where"] and detail["reason"] and detail["hint"]
    storage.aget_signed_url.assert_not_awaited()


async def test_the_organization_switch_turns_the_same_link_on(
    world, clock, in_cell, http, storage
):
    await set_org_links(world, True)
    await world.make_run(
        recording_url=RECORDING_PATH,
        public_access_token=OLD_LINK_ID,
        public_access_expires_at=T + timedelta(days=1),
    )
    assert (await open_link(http, OLD_LINK_ID)).status_code == 302


async def test_outside_a_cell_links_stay_on(world, clock, monkeypatch, http, storage):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    run = await world.make_run(recording_url=RECORDING_PATH)
    link_id = await db_client.ensure_public_access_token(run.id)
    assert link_id and (await open_link(http, link_id)).status_code == 302


# ---------------------------------------------------------------------------
# Links off: no link id, storage URL or storage path in any response or payload
# ---------------------------------------------------------------------------


def assert_clean(value):
    text = repr(value)
    for leak in (OLD_LINK_ID, CANARY_BUCKET, "/public/download/"):
        assert leak not in text, f"{leak!r} leaked in {text[:400]}"


async def stale_run(world):
    """A run that still holds a link id (minted while links were on) and every kind of recording."""
    return await world.make_run(
        recording_url=RECORDING_PATH,
        transcript_url=TRANSCRIPT_PATH,
        extra={
            "recordings": {
                "user": {"storage_key": USER_PATH, "storage_backend": "s3"},
                "bot": {"storage_key": BOT_PATH, "storage_backend": "s3"},
            }
        },
        public_access_token=OLD_LINK_ID,
        public_access_expires_at=T + timedelta(days=1),
    )


async def test_the_run_route_gives_no_link_and_no_public_url_when_links_are_off(
    world, clock, in_cell
):
    run = await stale_run(world)
    body = await get_workflow_run(world.workflow.id, run.id, user=world.user)

    assert body["public_access_token"] is None
    for field in PUBLIC_URL_FIELDS:
        assert body[field] is None, field
    assert OLD_LINK_ID not in repr(body)
    assert "/public/download/" not in repr(body)


async def test_the_run_route_does_not_mint_when_links_are_off(world, clock, in_cell):
    run = await world.make_run(recording_url=RECORDING_PATH)
    body = await get_workflow_run(world.workflow.id, run.id, user=world.user)
    assert body["public_access_token"] is None
    assert (await world.reload(run.id)).public_access_token is None


async def test_the_run_route_still_gives_the_links_when_they_are_on(
    world, clock, in_cell
):
    await set_org_links(world, True)
    run = await stale_run(world)
    body = await get_workflow_run(world.workflow.id, run.id, user=world.user)
    assert body["public_access_token"] == OLD_LINK_ID
    assert OLD_LINK_ID in body["recording_public_url"]


async def test_the_run_route_does_not_re_mint_a_revoked_run(world, clock):
    run = await world.make_run(recording_url=RECORDING_PATH)
    await db_client.ensure_public_access_token(run.id)
    await db_client.revoke_public_access(run.id)
    body = await get_workflow_run(world.workflow.id, run.id, user=world.user)
    assert body["public_access_token"] is None
    assert body["recording_public_url"] is None


async def test_the_usage_route_gives_no_link_and_no_public_url_when_links_are_off(
    world, clock, in_cell
):
    run = await stale_run(world)
    page = await get_usage_history(
        start_date=None,
        end_date=None,
        page=1,
        limit=50,
        filters=None,
        sort_by=None,
        sort_order="desc",
        user=world.user,
    )
    (row,) = [r for r in page["runs"] if r["id"] == run.id]
    assert row["public_access_token"] is None
    for field in PUBLIC_URL_FIELDS:
        assert row[field] is None, field
    assert OLD_LINK_ID not in repr(row)


def _run_for_context(extra):
    return SimpleNamespace(
        id=42,
        name="call",
        workflow_id=7,
        workflow=SimpleNamespace(name="bot"),
        campaign_id=None,
        created_at=None,
        initial_context={},
        gathered_context={},
        usage_info={},
        annotations={},
        extra=extra,
        recording_url=RECORDING_PATH,
        transcript_url=TRANSCRIPT_PATH,
    )


RECORDINGS_EXTRA = {
    "recordings": {
        "user": {"storage_key": USER_PATH},
        "bot": {"storage_key": BOT_PATH},
    },
    "note": "kept",
}

WEBHOOK_TEMPLATE = {
    "recording": "{{recording_url}}",
    "transcript": "{{transcript_url}}",
    "user": "{{user_recording_url}}",
    "bot": "{{bot_recording_url}}",
    "extra": "{{extra}}",
    "user_path": "{{extra.recordings.user.storage_key}}",
}


def test_the_webhook_payload_has_no_url_or_storage_path_without_a_link():
    context = _build_render_context(_run_for_context(RECORDINGS_EXTRA), None)
    payload = _build_webhook_payload(
        WebhookNodeData(
            name="hook",
            enabled=True,
            endpoint_url="https://example.com/hook",
            payload_template=WEBHOOK_TEMPLATE,
        ),
        context,
    )
    assert_clean(payload)
    assert_clean(context)
    # The rest of ``extra`` is untouched: only the recordings branch is internal.
    assert context["extra"] == {"note": "kept"}


def test_the_webhook_payload_still_carries_the_public_urls_with_a_link():
    with patch(
        "api.tasks.run_integrations.BACKEND_API_ENDPOINT", "https://api.example"
    ):
        context = _build_render_context(_run_for_context(RECORDINGS_EXTRA), NEW_LINK_ID)
    assert context["recording_url"].endswith(f"/{NEW_LINK_ID}/recording")
    assert context["user_recording_url"].endswith(f"/{NEW_LINK_ID}/user_recording")


def test_noveum_and_tuner_send_no_recording_url_without_a_link():
    context = SimpleNamespace(
        public_token=None, workflow_run=_run_for_context(RECORDINGS_EXTRA)
    )
    assert noveum_recording_url(context) is None
    assert tuner_recording_url(context) is None


def test_noveum_and_tuner_still_send_the_public_url_with_a_link():
    context = SimpleNamespace(
        public_token=NEW_LINK_ID, workflow_run=_run_for_context(RECORDINGS_EXTRA)
    )
    assert noveum_recording_url(context).endswith(f"/{NEW_LINK_ID}/recording")
    assert tuner_recording_url(context).endswith(f"/{NEW_LINK_ID}/recording")


def test_the_csv_report_has_empty_link_cells_without_a_link():
    run = SimpleNamespace(
        id=1,
        campaign_id=None,
        workflow_id=2,
        definition_id=None,
        created_at=T,
        initial_context={},
        gathered_context={},
        usage_info={},
        public_access_token=None,
    )
    report = build_run_report_csv([run]).getvalue()
    assert_clean(report)
    assert report.splitlines()[1].endswith(",,")
