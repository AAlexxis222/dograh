"""Concurrency slots on Redis's clock, two-phase lease and single release funnel (VOZ-AT-B3-21, VOZ-AT-B3-24).

Every test except the Redis-down ones runs against a real Redis (REDIS_URL): the behaviour under test lives in
the Lua scripts. ``fake_redis_clock`` makes the scripts read their "now" from a test-only key instead of TIME.
"""

import os
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from api.services.call_concurrency import CallConcurrencyLimitError, keys
from api.services.call_concurrency.rate_limiter import (
    FLEET_CONCURRENT_KEY,
    AdmissionBackendUnavailable,
    RateLimiter,
)
from api.services.call_concurrency.service import CallConcurrencyService
from api.services.runtime.durations import Durations

requires_redis = pytest.mark.skipif(
    "REDIS_URL" not in os.environ,
    reason="docker_missing: needs a real Redis (REDIS_URL via .env.test)",
)


@pytest.fixture
def org_id():
    return uuid.uuid4().int % 10_000_000


@pytest.fixture
async def cleanup(redis_client, org_id):
    """Every key a test may write for its org, its scope and its runs; fleet members by attempt."""
    attempts: list[str] = []
    runs: list[int] = []
    yield SimpleNamespace(attempts=attempts, runs=runs)
    if attempts:
        await redis_client.zrem(FLEET_CONCURRENT_KEY, *attempts)
    await redis_client.delete(
        keys.org_key(org_id),
        keys.scope_key(f"campaign:{org_id}"),
        *(keys.mapping_key(run) for run in runs),
    )


async def _close(*limiters: RateLimiter) -> None:
    for limiter in limiters:
        await limiter.close()


@requires_redis
async def test_two_processes_with_skewed_clocks_count_exactly(
    monkeypatch, org_id, cleanup
):
    """VOZ-AT-B3-21: process clocks skewed by +-5 min must not move any count (scores come from Redis TIME)."""
    real_now = time.time()
    a, b = RateLimiter(), RateLimiter()

    def as_process(offset: float) -> None:
        monkeypatch.setattr(time, "time", lambda: real_now + offset)

    cleanup.attempts += ["att-a", "att-b", "att-c"]
    try:
        as_process(+300)
        baseline = await a.get_fleet_concurrent_count()
        assert await a.acquire_slot(org_id=org_id, attempt_id="att-a", max_concurrent=2)
        as_process(-300)
        assert await b.acquire_slot(org_id=org_id, attempt_id="att-b", max_concurrent=2)
        as_process(+300)
        assert await a.get_concurrent_count(org_id) == 2
        assert await a.get_fleet_concurrent_count() == baseline + 2
        as_process(-300)
        assert (
            await b.acquire_slot(org_id=org_id, attempt_id="att-c", max_concurrent=2)
            is None
        )
        as_process(+300)
        assert await a.release_slot(org_id=org_id, attempt_id="att-a") is True
        as_process(-300)
        assert await b.get_concurrent_count(org_id) == 1
        assert await b.get_fleet_concurrent_count() == baseline + 1
    finally:
        await _close(a, b)


@requires_redis
async def test_long_call_keeps_counter_exact(
    redis_client, fake_redis_clock, org_id, cleanup
):
    """A claimed call renewed per call stays counted past slot_ttl; the claim alone outlives pending_ttl_s."""
    durations = Durations.from_cell(7200)
    rl = RateLimiter(durations=durations, test_clock=fake_redis_clock.key)
    run_id = org_id
    cleanup.attempts.append("long")
    cleanup.runs.append(run_id)
    try:
        await rl.acquire_slot(org_id=org_id, attempt_id="long", max_concurrent=5)
        assert await rl.store_workflow_slot_mapping_if_absent(run_id, org_id, "long")
        lease = await rl.claim_slot(run_id)
        assert lease and lease.attempt_id == "long" and lease.org_id == org_id
        await fake_redis_clock.advance(durations.pending_ttl_s + 1)
        assert await rl.get_concurrent_count(org_id) == 1  # claimed: no longer pending
        for _ in range(
            (durations.slot_ttl + 500) // 100
        ):  # renew every 100 s past slot_ttl
            await fake_redis_clock.advance(100)
            await rl.renew_slot(
                org_id=org_id, attempt_id="long", workflow_run_id=run_id
            )
        assert await rl.get_concurrent_count(org_id) == 1
        assert await rl.get_fleet_concurrent_count() >= 1
        assert await redis_client.zscore(FLEET_CONCURRENT_KEY, "long") > (
            fake_redis_clock.now
        )
    finally:
        await _close(rl)


@requires_redis
async def test_pending_lease_expires_if_never_claimed(
    redis_client, fake_redis_clock, org_id, cleanup
):
    """VOZ-AT-B3-24: an admission no worker claims frees the org and fleet counts after pending_ttl_s."""
    durations = Durations.from_cell(1200)
    rl = RateLimiter(durations=durations, test_clock=fake_redis_clock.key)
    cleanup.attempts.append("ghost")
    try:
        assert await rl.acquire_slot(
            org_id=org_id, attempt_id="ghost", max_concurrent=1
        )
        await fake_redis_clock.advance(durations.pending_ttl_s - 1)
        assert await rl.get_concurrent_count(org_id) == 1
        await fake_redis_clock.advance(2)
        assert await rl.get_concurrent_count(org_id) == 0
        assert await redis_client.zscore(FLEET_CONCURRENT_KEY, "ghost") <= (
            fake_redis_clock.now
        )  # the fleet reader skips it: its score is an expiry in the past
        assert await rl.acquire_slot(org_id=org_id, attempt_id="next", max_concurrent=1)
        cleanup.attempts.append("next")
    finally:
        await _close(rl)


@requires_redis
async def test_acquire_release_round_trip_uses_the_canonical_member_on_every_key(
    redis_client, org_id, cleanup
):
    """VOZ-AC-B3-32: one member format (attempt_id) on org, scope and fleet; one release clears all + mapping."""
    rl = RateLimiter()
    scope = f"campaign:{org_id}"
    attempt = keys.new_attempt_id()
    run_id = org_id
    cleanup.attempts.append(attempt)
    cleanup.runs.append(run_id)
    try:
        acquired = await rl.acquire_slot(
            org_id=org_id,
            attempt_id=attempt,
            max_concurrent=5,
            scope_key=scope,
            scope_max_concurrent=5,
        )
        assert acquired and acquired.slot_id == attempt
        await rl.store_workflow_slot_mapping_if_absent(
            run_id, org_id, attempt, scope_key=scope
        )
        for key in (keys.org_key(org_id), keys.scope_key(scope), FLEET_CONCURRENT_KEY):
            assert await redis_client.zscore(key, attempt) is not None, key

        assert (
            await rl.release_slot(
                org_id=org_id,
                attempt_id=attempt,
                scope_key=scope,
                workflow_run_id=run_id,
            )
            is True
        )
        for key in (keys.org_key(org_id), keys.scope_key(scope), FLEET_CONCURRENT_KEY):
            assert await redis_client.zscore(key, attempt) is None, key
        assert not await redis_client.exists(keys.mapping_key(run_id))
    finally:
        await _close(rl)


@requires_redis
async def test_release_is_idempotent_and_keeps_a_mapping_it_does_not_own(
    redis_client, org_id, cleanup
):
    rl = RateLimiter()
    run_id = org_id
    cleanup.attempts += ["owner", "other"]
    cleanup.runs.append(run_id)
    try:
        await rl.acquire_slot(org_id=org_id, attempt_id="owner", max_concurrent=5)
        await rl.acquire_slot(org_id=org_id, attempt_id="other", max_concurrent=5)
        await rl.store_workflow_slot_mapping_if_absent(run_id, org_id, "owner")

        # Releasing another attempt with the run's mapping key leaves the owner's mapping alone.
        assert await rl.release_slot(
            org_id=org_id, attempt_id="other", workflow_run_id=run_id
        )
        assert await rl.get_workflow_slot_mapping(run_id) == (org_id, "owner", None)

        assert await rl.release_slot(
            org_id=org_id, attempt_id="owner", workflow_run_id=run_id
        )
        # A mapping left behind by a slot that already expired still goes with the (now idempotent) release.
        await rl.store_workflow_slot_mapping_if_absent(run_id, org_id, "owner")
        again = await rl.release_slot(
            org_id=org_id, attempt_id="owner", workflow_run_id=run_id
        )
        assert again is False
        assert await rl.get_concurrent_count(org_id) == 0
        assert not await redis_client.exists(keys.mapping_key(run_id))
    finally:
        await _close(rl)


@requires_redis
async def test_release_clears_semaphore_members_registered_in_the_mapping(
    redis_client, org_id, cleanup
):
    """The seam for the provider semaphores: `sem:<key>` fields of the mapping are released with the slot."""
    rl = RateLimiter()
    run_id = org_id
    semaphore = f"test_semaphore:{org_id}"
    cleanup.attempts.append("with-sem")
    cleanup.runs.append(run_id)
    try:
        await rl.acquire_slot(org_id=org_id, attempt_id="with-sem", max_concurrent=5)
        await rl.store_workflow_slot_mapping_if_absent(run_id, org_id, "with-sem")
        await redis_client.zadd(semaphore, {"with-sem": time.time() + 100})
        await redis_client.hset(
            keys.mapping_key(run_id), f"sem:{semaphore}", "with-sem"
        )

        assert await rl.release_slot(
            org_id=org_id, attempt_id="with-sem", workflow_run_id=run_id
        )
        assert await redis_client.zscore(semaphore, "with-sem") is None
    finally:
        await redis_client.delete(semaphore)
        await _close(rl)


@requires_redis
async def test_renewal_re_adds_the_slot_and_mapping_after_their_keys_were_lost(
    redis_client, org_id, cleanup
):
    """VOZ-AC-B3-65-bis / -66: ZADD without XX and the mapping re-created, so a Redis restart heals itself."""
    rl = RateLimiter()
    scope = f"campaign:{org_id}"
    run_id = org_id
    cleanup.attempts.append("survivor")
    cleanup.runs.append(run_id)
    try:
        await rl.acquire_slot(
            org_id=org_id,
            attempt_id="survivor",
            max_concurrent=5,
            scope_key=scope,
            scope_max_concurrent=5,
        )
        await rl.store_workflow_slot_mapping_if_absent(
            run_id, org_id, "survivor", scope_key=scope
        )
        await redis_client.delete(
            keys.org_key(org_id), keys.scope_key(scope), keys.mapping_key(run_id)
        )
        await redis_client.zrem(FLEET_CONCURRENT_KEY, "survivor")

        await rl.renew_slot(
            org_id=org_id,
            attempt_id="survivor",
            scope_key=scope,
            workflow_run_id=run_id,
        )

        for key in (keys.org_key(org_id), keys.scope_key(scope), FLEET_CONCURRENT_KEY):
            assert await redis_client.zscore(key, "survivor") is not None, key
        assert await rl.get_workflow_slot_mapping(run_id) == (org_id, "survivor", scope)
        slot_ttl = rl.durations.slot_ttl
        for key in (
            keys.org_key(org_id),
            keys.scope_key(scope),
            keys.mapping_key(run_id),
        ):
            assert slot_ttl - 5 <= await redis_client.ttl(key) <= slot_ttl, key
    finally:
        await _close(rl)


@requires_redis
async def test_terminal_status_callback_releases_through_the_funnel(
    redis_client, org_id, cleanup
):
    """VOZ-AT-B3-24: the carrier's terminal callback frees slot and mapping (status_processor -> release funnel)."""
    from api.services.call_concurrency.rate_limiter import rate_limiter
    from api.services.telephony.status_processor import (
        StatusCallbackRequest,
        _process_status_update,
    )

    run_id = org_id
    attempt = keys.new_attempt_id()
    cleanup.attempts.append(attempt)
    cleanup.runs.append(run_id)
    await rate_limiter.acquire_slot(org_id=org_id, attempt_id=attempt, max_concurrent=5)
    await rate_limiter.store_workflow_slot_mapping_if_absent(run_id, org_id, attempt)

    db = AsyncMock()
    db.get_workflow_run_by_id.return_value = SimpleNamespace(
        logs={}, campaign_id=None, state="completed"
    )
    with patch("api.services.telephony.status_processor.db_client", db):
        await _process_status_update(
            run_id, StatusCallbackRequest(call_id="c-1", status="completed")
        )

    assert await rate_limiter.get_concurrent_count(org_id) == 0
    assert await redis_client.zscore(FLEET_CONCURRENT_KEY, attempt) is None
    assert not await redis_client.exists(keys.mapping_key(run_id))


@requires_redis
async def test_call_slot_hold_claims_on_entry_and_releases_on_exit(
    redis_client, org_id, cleanup
):
    """The per-call seam of the worker: claim when the call starts, renewal task gone and slot released at the end."""
    from api.services.call_concurrency.rate_limiter import rate_limiter

    service = CallConcurrencyService()
    run_id = org_id
    attempt = keys.new_attempt_id()
    cleanup.attempts.append(attempt)
    cleanup.runs.append(run_id)
    await rate_limiter.acquire_slot(org_id=org_id, attempt_id=attempt, max_concurrent=5)
    await rate_limiter.store_workflow_slot_mapping_if_absent(run_id, org_id, attempt)
    pending_score = await redis_client.zscore(keys.org_key(org_id), attempt)

    async with service.hold_run_slot(run_id):
        claimed_score = await redis_client.zscore(keys.org_key(org_id), attempt)
        assert claimed_score - pending_score > rate_limiter.durations.pending_ttl_s

    assert await redis_client.zscore(keys.org_key(org_id), attempt) is None
    assert not await redis_client.exists(keys.mapping_key(run_id))


async def test_acquire_fails_closed_when_redis_is_down():
    """VOZ-AC-B3-65-bis: a backend error is not a free slot and not 'limit reached'."""
    rl = RateLimiter()
    client = AsyncMock()
    client.eval.side_effect = RedisConnectionError("redis down")
    rl._get_redis = AsyncMock(return_value=client)

    with pytest.raises(AdmissionBackendUnavailable) as e:
        await rl.acquire_slot(org_id=1, attempt_id="a", max_concurrent=5)
    assert e.value.reason == "admission_backend_unavailable" and e.value.hint


async def test_service_rejects_with_the_backend_reason_without_waiting_or_usage_event():
    service = CallConcurrencyService()
    with (
        patch("api.services.call_concurrency.service.db_client") as db,
        patch("api.services.call_concurrency.service.rate_limiter") as rl,
        patch.object(service, "_notify_limit_reached", AsyncMock()) as notify,
    ):
        db.get_configuration = AsyncMock(return_value=None)
        rl.acquire_slot = AsyncMock(
            side_effect=AdmissionBackendUnavailable("redis down")
        )
        with pytest.raises(CallConcurrencyLimitError) as e:
            await service.acquire_org_slot(1, source="inbound:twilio", timeout=30)

    assert e.value.reason == "admission_backend_unavailable"
    rl.acquire_slot.assert_awaited_once()  # fails closed at once: no 30 s of retries
    notify.assert_not_awaited()  # not a usage-limit event
