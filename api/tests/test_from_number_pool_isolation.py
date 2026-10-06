"""Caller-ID rotation and dial-rate buckets stay isolated across accounts/configs."""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.services.call_concurrency import CallConcurrencySlot
from api.services.call_concurrency.rate_limiter import RateLimiter
from api.services.campaign.campaign_call_dispatcher import CampaignCallDispatcher


@pytest.fixture
async def limiter():
    rl = RateLimiter()
    redis = await rl._get_redis()
    keys = set()
    original = redis.eval

    async def track(script, count, *args):
        keys.update(args[:count])
        return await original(script, count, *args)

    redis.eval = track
    yield rl
    redis.eval = original
    if keys:
        await redis.delete(*keys)
    await rl.close()


@pytest.mark.asyncio
async def test_rotation_reuses_numbers_without_reservations_and_isolates_configs(
    limiter,
):
    org = uuid.uuid4().int % 10**9
    numbers = ["cli-b", "cli-a", "cli-a"]
    assert [await limiter.select_from_number(org, 1, numbers) for _ in range(5)] == [
        "cli-a",
        "cli-b",
        "cli-a",
        "cli-b",
        "cli-a",
    ]
    assert await limiter.select_from_number(org, 2, ["other-cli"]) == "other-cli"
    assert await limiter.select_from_number(org + 1, 1, numbers) == "cli-a"
    # Removed addresses must not survive in a Redis pool; selection uses this snapshot.
    assert await limiter.select_from_number(org, 1, ["new-cli"]) == "new-cli"


@pytest.mark.asyncio
async def test_parallel_rotation_balances_without_exhausting(limiter):
    org = uuid.uuid4().int % 10**9
    selected = await asyncio.gather(
        *[limiter.select_from_number(org, 1, ["a", "b"]) for _ in range(200)]
    )
    assert selected.count("a") == selected.count("b") == 100


@pytest.mark.asyncio
async def test_rotation_failure_falls_back_and_empty_list_remains_optional():
    rl = RateLimiter()
    with patch.object(
        rl, "_get_redis", AsyncMock(side_effect=ConnectionError("offline"))
    ):
        assert await rl.select_from_number(1, 1, ["b", "a"]) == "a"
        assert await rl.select_from_number(1, 1, []) is None


@pytest.mark.asyncio
async def test_campaign_buckets_and_waits_are_independent(limiter):
    org = uuid.uuid4().int % 10**9
    scope_a, scope_b = f"test:campaign:{org}:a", f"test:campaign:{org}:b"
    assert await limiter.acquire_token(org, 1, scope_key=scope_a)
    assert not await limiter.acquire_token(org, 1, scope_key=scope_a)
    assert 0 < await limiter.get_next_available_slot(org, 1, scope_key=scope_a) <= 1
    assert await limiter.get_next_available_slot(org, 4, scope_key=scope_b) == 0
    assert all(
        [await limiter.acquire_token(org, 4, scope_key=scope_b) for _ in range(4)]
    )
    assert not await limiter.acquire_token(org, 4, scope_key=scope_b)
    await asyncio.sleep(1.01)
    assert await limiter.acquire_token(org, 1, scope_key=scope_a)


@pytest.mark.asyncio
async def test_concurrent_tokens_with_identical_timestamps_are_counted_separately(
    limiter,
):
    org = uuid.uuid4().int % 10**9
    with patch(
        "api.services.call_concurrency.rate_limiter.time.time", return_value=1234567890
    ):
        admitted = await asyncio.gather(
            *[
                limiter.acquire_token(org, 4, scope_key=f"test:campaign:{org}")
                for _ in range(20)
            ]
        )
    assert sum(admitted) == 4


@pytest.mark.asyncio
async def test_two_caller_ids_can_fill_200_real_slots_without_exceeding_org_limit():
    rl = RateLimiter()
    org = uuid.uuid4().int % 10**9
    scope = f"test:campaign:{org}"
    slots = []
    numbers = []
    try:
        for _ in range(200):
            slot = await rl.try_acquire_concurrent_slot_details(
                org,
                200,
                scope_key=scope,
                scope_max_concurrent=200,
            )
            assert slot is not None
            slots.append(slot)
            numbers.append(await rl.select_from_number(org, 55, ["a", "b"]))
        assert len(set(numbers)) == 2
        assert await rl.get_concurrent_count(org) == 200
        assert await rl.try_acquire_concurrent_slot_details(org, 200) is None
    finally:
        # Only release this test's fleet members; never delete the shared fleet set.
        for slot in slots:
            await rl.release_concurrent_slot(org, slot.slot_id, scope_key=scope)
        redis = await rl._get_redis()
        await redis.delete(
            f"concurrent_calls:{org}",
            f"concurrent_calls:{scope}",
            f"caller_id_rotation:{org}:55",
        )
        await rl.close()


# ---------------------------------------------------------------------------
# Dispatcher: must thread telephony_configuration_id end-to-end
# ---------------------------------------------------------------------------


def _make_campaign(
    *,
    organization_id: int,
    telephony_configuration_id: int | None,
    workflow_id: int = 1,
    campaign_id: int = 99,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=campaign_id,
        organization_id=organization_id,
        workflow_id=workflow_id,
        created_by=1,
        telephony_configuration_id=telephony_configuration_id,
        rate_limit_per_second=100,
        processed_rows=0,
        orchestrator_metadata={},
    )


def _make_queued_run(
    *,
    queued_run_id: int = 1,
    phone_number: str = "+15559990001",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=queued_run_id,
        source_uuid=f"src-{queued_run_id}",
        context_variables={"phone_number": phone_number},
    )


def _make_slot(org_id: int) -> CallConcurrencySlot:
    return CallConcurrencySlot(
        organization_id=org_id,
        slot_id="slot-1",
        max_concurrent=1,
        source="test",
    )


def _mock_db(org_id: int) -> MagicMock:
    db = MagicMock()
    db.get_workflow = AsyncMock(
        return_value=SimpleNamespace(
            id=1,
            organization_id=org_id,
            released_definition=SimpleNamespace(id=55),
            current_definition=None,
        )
    )
    db.get_definition_configurations_with_owner = AsyncMock(return_value=({}, org_id))
    db.get_configuration_value = AsyncMock(return_value={})
    db.create_workflow_run = AsyncMock(return_value=SimpleNamespace(id=555, logs={}))
    db.update_workflow_run = AsyncMock()
    db.update_queued_run = AsyncMock()
    return db


class TestDispatcherThreadsTelephonyConfig:
    """The dispatcher must rotate caller IDs within the campaign's
    telephony_configuration_id and pin legacy campaigns to a ready config."""

    @pytest.mark.asyncio
    async def test_legacy_campaign_pins_the_ready_resolved_config(self):
        campaign = _make_campaign(
            organization_id=7,
            telephony_configuration_id=None,
        )
        provider = MagicMock()
        resolver = AsyncMock(return_value=4242)
        provider_factory = AsyncMock(return_value=provider)
        dispatcher = CampaignCallDispatcher()

        with (
            patch(
                "api.services.telephony.outbound_readiness.resolve_outbound_configuration_id",
                resolver,
            ),
            patch(
                "api.services.telephony.factory.get_telephony_provider_by_id",
                provider_factory,
            ),
        ):
            result = await dispatcher.get_provider_for_campaign(campaign)

        assert result is provider
        assert campaign.telephony_configuration_id == 4242
        assert resolver.await_args.args == (None, 7)
        provider_factory.assert_awaited_once_with(4242, 7)

    @pytest.mark.asyncio
    async def test_dispatch_call_selects_from_number_for_campaign_config(self):
        org_id = 7
        config_id = 4242
        campaign = _make_campaign(
            organization_id=org_id, telephony_configuration_id=config_id
        )
        queued_run = _make_queued_run()

        provider = MagicMock()
        provider.PROVIDER_NAME = "twilio"
        provider.WEBHOOK_ENDPOINT = "twilio/voice"
        provider.from_numbers = ["+15551110001"]
        provider.initiate_call = AsyncMock(
            return_value=SimpleNamespace(call_id="call-1", provider_metadata={})
        )
        mock_db = _mock_db(org_id)
        dispatcher = CampaignCallDispatcher()

        with (
            patch.object(
                dispatcher,
                "get_provider_for_campaign",
                AsyncMock(return_value=provider),
            ),
            patch("api.services.campaign.campaign_call_dispatcher.db_client", mock_db),
            patch(
                "api.services.campaign.campaign_call_dispatcher.rate_limiter"
            ) as mock_rl,
            patch(
                "api.services.campaign.campaign_call_dispatcher.call_concurrency"
            ) as mock_concurrency,
            patch(
                "api.services.campaign.campaign_call_dispatcher.get_backend_endpoints",
                AsyncMock(return_value=("https://example.com", None)),
            ),
            patch(
                "api.services.campaign.campaign_call_dispatcher.authorize_workflow_run_start",
                AsyncMock(
                    return_value=SimpleNamespace(has_quota=True, error_message="")
                ),
            ),
        ):
            mock_concurrency.bind_workflow_run = AsyncMock()
            mock_concurrency.release_slot = AsyncMock()
            mock_concurrency.release_workflow_run_slot = AsyncMock()
            mock_rl.select_from_number = AsyncMock(return_value="+15551110001")
            mock_rl.acquire_token = AsyncMock(return_value=True)

            await dispatcher.dispatch_call(queued_run, campaign, _make_slot(org_id))

        mock_db.get_workflow.assert_awaited_once_with(
            campaign.workflow_id,
            organization_id=org_id,
        )
        mock_rl.select_from_number.assert_awaited_once_with(
            org_id, config_id, provider.from_numbers
        )
        assert provider.initiate_call.await_count == 1
        call_kwargs = provider.initiate_call.await_args.kwargs
        assert call_kwargs["from_number"] == "+15551110001"
        assert "campaign_id=" not in call_kwargs["webhook_url"], (
            "campaign outbound answer_url should not include campaign_id; "
            f"got {call_kwargs['webhook_url']}"
        )

    @pytest.mark.asyncio
    async def test_dispatch_rejects_incomplete_setup_before_creating_run(self):
        from api.services.telephony.outbound_readiness import (
            OutboundSetupIncompleteError,
        )

        org_id = 7
        config_id = 4242
        campaign = _make_campaign(
            organization_id=org_id, telephony_configuration_id=config_id
        )
        queued_run = _make_queued_run()
        mock_db = _mock_db(org_id)
        slot = _make_slot(org_id)
        dispatcher = CampaignCallDispatcher()

        with (
            patch.object(
                dispatcher,
                "get_provider_for_campaign",
                AsyncMock(
                    side_effect=OutboundSetupIncompleteError(
                        config_id,
                        "Twilio campaign",
                        "Add a caller ID before placing calls.",
                    )
                ),
            ),
            patch("api.services.campaign.campaign_call_dispatcher.db_client", mock_db),
            patch(
                "api.services.campaign.campaign_call_dispatcher.call_concurrency"
            ) as mock_concurrency,
        ):
            mock_concurrency.release_slot = AsyncMock()
            mock_concurrency.release_workflow_run_slot = AsyncMock()
            with pytest.raises(OutboundSetupIncompleteError):
                await dispatcher.dispatch_call(queued_run, campaign, slot)

        mock_db.create_workflow_run.assert_not_awaited()
        mock_concurrency.release_slot.assert_awaited_once_with(slot)
