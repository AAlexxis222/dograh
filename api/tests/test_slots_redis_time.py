"""Concurrency slots on Redis's clock, two-phase lease and single release funnel (VOZ-AT-B3-21, VOZ-AT-B3-24).

Every test except the Redis-down ones runs against a real Redis (REDIS_URL): the behaviour under test lives in
the Lua scripts. ``fake_redis_clock`` makes the scripts read their "now" from a test-only key instead of TIME.
"""

import asyncio
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger
from redis.exceptions import ConnectionError as RedisConnectionError

from api.services.call_concurrency import CallConcurrencyLimitError, keys
from api.services.call_concurrency.rate_limiter import (
    FLEET_CONCURRENT_KEY,
    LATE_CLAIM_OVERCOMMIT,
    AdmissionBackendUnavailable,
    RateLimiter,
)
from api.services.call_concurrency.service import CallConcurrencyService
from api.services.runtime.durations import Durations


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
        await redis_client.zrem(keys.fleet_key(), *attempts)
        await redis_client.zrem(
            FLEET_CONCURRENT_KEY,
            *(keys.legacy_fleet_member(org_id, a) for a in attempts),
        )
    await redis_client.delete(
        keys.org_key(org_id),
        keys.scope_key(f"campaign:{org_id}"),
        keys.legacy_org_key(org_id),
        keys.legacy_scope_key(f"campaign:{org_id}"),
        *(keys.mapping_key(run) for run in runs),
    )


async def _close(*limiters: RateLimiter) -> None:
    for limiter in limiters:
        await limiter.close()


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
        assert await redis_client.zscore(keys.fleet_key(), "long") > (
            fake_redis_clock.now
        )
    finally:
        await _close(rl)


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
        assert await redis_client.zscore(keys.fleet_key(), "ghost") <= (
            fake_redis_clock.now
        )  # the fleet reader skips it: its score is an expiry in the past
        assert await rl.acquire_slot(org_id=org_id, attempt_id="next", max_concurrent=1)
        cleanup.attempts.append("next")
    finally:
        await _close(rl)


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
        for key in (keys.org_key(org_id), keys.scope_key(scope), keys.fleet_key()):
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
        for key in (keys.org_key(org_id), keys.scope_key(scope), keys.fleet_key()):
            assert await redis_client.zscore(key, attempt) is None, key
        assert not await redis_client.exists(keys.mapping_key(run_id))
    finally:
        await _close(rl)


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
        await redis_client.zrem(keys.fleet_key(), "survivor")

        await rl.renew_slot(
            org_id=org_id,
            attempt_id="survivor",
            scope_key=scope,
            workflow_run_id=run_id,
        )

        for key in (keys.org_key(org_id), keys.scope_key(scope), keys.fleet_key()):
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


@pytest.mark.parametrize(
    "status", ["completed", "no-answer", "busy", "failed", "canceled", "error"]
)
async def test_terminal_status_callback_releases_through_the_funnel(
    redis_client, org_id, cleanup, status
):
    """VOZ-AT-B3-24: every terminal carrier callback (answered or never connected) frees the slot and the mapping
    through the release funnel at once, without waiting for the pending lease to run out."""
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
        logs={},
        campaign_id=None,
        state="completed",
        is_completed=True,
        gathered_context={},
        workflow=SimpleNamespace(organization_id=org_id),
    )
    with (
        patch("api.services.telephony.status_processor.db_client", db),
        patch(
            "api.services.telephony.status_processor.map_disposition",
            AsyncMock(return_value=status),
        ),
    ):
        await _process_status_update(
            run_id, StatusCallbackRequest(call_id="c-1", status=status)
        )

    assert await rate_limiter.get_concurrent_count(org_id) == 0
    assert await redis_client.zscore(keys.fleet_key(), attempt) is None
    assert not await redis_client.exists(keys.mapping_key(run_id))


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
    assert "admission_backend_unavailable" in str(e.value) and "hint:" in str(e.value)
    rl.acquire_slot.assert_awaited_once()  # fails closed at once: no 30 s of retries
    notify.assert_not_awaited()  # not a usage-limit event


# ---------------------------------------------------------------------------
# Fix round 1: outbound ringing, late claims, legacy keys, the renewal task itself.
# ---------------------------------------------------------------------------


@pytest.fixture
def warnings_log():
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    yield records
    logger.remove(sink)


async def _admit_and_bind(rl, org_id, attempt, run_id, max_concurrent, **kwargs):
    assert await rl.acquire_slot(
        org_id=org_id, attempt_id=attempt, max_concurrent=max_concurrent, **kwargs
    )
    assert await rl.store_workflow_slot_mapping_if_absent(
        run_id, org_id, attempt, max_concurrent=max_concurrent
    )


async def _wait_for_score(redis_client, key, member, at_least, tries=60):
    score = None
    for _ in range(tries):
        score = await redis_client.zscore(key, member)
        if score is not None and score >= at_least:
            break
        await asyncio.sleep(0.05)
    return score


async def test_outbound_admission_keeps_its_slot_while_the_callee_rings(
    fake_redis_clock, org_id, cleanup
):
    """Finding 1 / probe P1: an outbound call reaches a worker only once answered, so 35 s of ringing must not
    free its slot; one that is never answered still frees it after the ringing bound."""
    d = Durations.from_cell(1200)
    assert d.outbound_pending_ttl_s == d.ring_timeout_s + d.pending_ttl_s
    rl = RateLimiter(durations=d, test_clock=fake_redis_clock.key)
    cleanup.attempts += ["ring-a", "ring-b"]
    try:
        assert await rl.acquire_slot(
            org_id=org_id, attempt_id="ring-a", max_concurrent=1, outbound=True
        )
        await fake_redis_clock.advance(35)
        assert (
            await rl.acquire_slot(org_id=org_id, attempt_id="ring-b", max_concurrent=1)
            is None
        )
        await fake_redis_clock.advance(d.outbound_pending_ttl_s - 35 - 1)
        assert await rl.get_concurrent_count(org_id) == 1
        await fake_redis_clock.advance(2)
        assert await rl.get_concurrent_count(org_id) == 0
    finally:
        await _close(rl)


async def test_late_claim_within_the_limit_is_re_added_without_an_overcommit(
    fake_redis_clock, org_id, cleanup, warnings_log
):
    rl = RateLimiter(test_clock=fake_redis_clock.key)
    run_id = org_id
    cleanup.attempts.append("late")
    cleanup.runs.append(run_id)
    try:
        await _admit_and_bind(rl, org_id, "late", run_id, 2)
        await fake_redis_clock.advance(rl.durations.pending_ttl_s + 1)
        assert await rl.get_concurrent_count(org_id) == 0
        assert await rl.claim_slot(run_id)
        assert await rl.get_concurrent_count(org_id) == 1
        assert not [m for m in warnings_log if LATE_CLAIM_OVERCOMMIT in m]
    finally:
        await _close(rl)


async def test_late_claim_over_a_full_limit_keeps_the_call_and_names_the_overcommit(
    fake_redis_clock, org_id, cleanup, warnings_log
):
    """Finding 1(c) / probe P1: never drop an answered call, but count it and say so."""
    rl = RateLimiter(test_clock=fake_redis_clock.key)
    run_id = org_id
    cleanup.attempts += ["ring-a", "ring-b"]
    cleanup.runs.append(run_id)
    try:
        await _admit_and_bind(rl, org_id, "ring-a", run_id, 1)
        await fake_redis_clock.advance(rl.durations.pending_ttl_s + 5)
        assert await rl.acquire_slot(
            org_id=org_id, attempt_id="ring-b", max_concurrent=1
        )
        assert await rl.claim_slot(run_id)  # ring-a answers after its lease expired
        assert await rl.get_concurrent_count(org_id) == 2
        [event] = [m for m in warnings_log if LATE_CLAIM_OVERCOMMIT in m]
        assert f"org={org_id}" in event and "attempt_id=ring-a" in event
        assert "count=2" in event
    finally:
        await _close(rl)


async def test_legacy_calls_still_count_toward_every_limit(
    redis_client, fake_redis_clock, org_id, cleanup
):
    """Finding 2 / probe P2: calls the previous code admitted (score = acquisition time, un-versioned keys) keep
    counting during a rolling deploy, under the old rule (live while score > now - slot_ttl)."""
    rl = RateLimiter(test_clock=fake_redis_clock.key)
    now, slot_ttl = fake_redis_clock.now, rl.durations.slot_ttl
    scope = f"campaign:{org_id}"
    cleanup.attempts += ["old-1", "new-1", "new-2"]
    try:
        fleet_before = await rl.get_fleet_concurrent_count()
        await redis_client.zadd(
            keys.legacy_org_key(org_id),
            {"old-1": now - 5, "old-2": now - 600, "old-dead": now - slot_ttl - 1},
        )
        await redis_client.zadd(keys.legacy_scope_key(scope), {"old-1": now - 5})
        await redis_client.zadd(
            FLEET_CONCURRENT_KEY, {keys.legacy_fleet_member(org_id, "old-1"): now - 5}
        )

        assert await rl.get_concurrent_count(org_id) == 2
        assert (
            await rl.acquire_slot(org_id=org_id, attempt_id="new-1", max_concurrent=2)
            is None
        )
        assert (
            await rl.acquire_slot(
                org_id=org_id,
                attempt_id="new-1",
                max_concurrent=5,
                scope_key=scope,
                scope_max_concurrent=1,
            )
            is None
        )
        assert await rl.acquire_slot(
            org_id=org_id, attempt_id="new-2", max_concurrent=3
        )
        assert await rl.get_fleet_concurrent_count() == fleet_before + 2
    finally:
        await _close(rl)


async def test_a_legacy_release_leaves_v2_alone_and_the_funnel_ends_legacy_calls(
    redis_client, fake_redis_clock, org_id, cleanup
):
    rl = RateLimiter(test_clock=fake_redis_clock.key)
    now = fake_redis_clock.now
    run_id = org_id
    cleanup.attempts += ["new-1", "old-1"]
    cleanup.runs.append(run_id)
    try:
        await rl.acquire_slot(org_id=org_id, attempt_id="new-1", max_concurrent=5)
        # The previous code's release: ZREM of its own member formats on its own keys.
        await redis_client.zrem(keys.legacy_org_key(org_id), "new-1")
        await redis_client.zrem(
            FLEET_CONCURRENT_KEY, keys.legacy_fleet_member(org_id, "new-1")
        )
        assert await rl.get_concurrent_count(org_id) == 1
        assert await redis_client.zscore(keys.fleet_key(), "new-1") is not None

        # A call the previous code admitted, taken and ended by the new code.
        legacy_member = keys.legacy_fleet_member(org_id, "old-1")
        await redis_client.zadd(keys.legacy_org_key(org_id), {"old-1": now - 5})
        await redis_client.zadd(FLEET_CONCURRENT_KEY, {legacy_member: now - 5})
        await rl.store_workflow_slot_mapping_if_absent(run_id, org_id, "old-1")
        assert await rl.claim_slot(run_id)
        await rl.renew_slot(org_id=org_id, attempt_id="old-1", workflow_run_id=run_id)
        assert await redis_client.zscore(keys.org_key(org_id), "old-1") is None
        assert await rl.get_concurrent_count(org_id) == 2  # counted once, as legacy

        assert await rl.release_slot(
            org_id=org_id, attempt_id="old-1", workflow_run_id=run_id
        )
        assert await redis_client.zscore(keys.legacy_org_key(org_id), "old-1") is None
        assert await redis_client.zscore(FLEET_CONCURRENT_KEY, legacy_member) is None
        assert await rl.get_concurrent_count(org_id) == 1
    finally:
        await _close(rl)


async def test_the_per_call_renewal_task_keeps_a_long_call_counted(
    monkeypatch, redis_client, fake_redis_clock, org_id, cleanup
):
    """Finding 3: the renewal task of hold_run_slot (not renew_slot called by hand) moves the score forward at
    every tick, past slot_ttl, and the call stays counted."""
    from api.services.call_concurrency import service as service_module

    rl = RateLimiter(test_clock=fake_redis_clock.key)
    monkeypatch.setattr(service_module, "rate_limiter", rl)
    run_id = org_id
    cleanup.attempts.append("long")
    cleanup.runs.append(run_id)
    slot_ttl = rl.durations.slot_ttl
    try:
        await _admit_and_bind(rl, org_id, "long", run_id, 5)
        start = fake_redis_clock.now
        async with CallConcurrencyService().hold_run_slot(
            run_id, renew_interval_s=0.01
        ):
            while fake_redis_clock.now - start <= slot_ttl + 400:
                await fake_redis_clock.advance(200)
                expected = fake_redis_clock.now + slot_ttl
                score = await _wait_for_score(
                    redis_client, keys.org_key(org_id), "long", expected
                )
                assert score >= expected, "the renewal task did not renew"
            assert await rl.get_concurrent_count(org_id) == 1
        assert await rl.get_concurrent_count(org_id) == 0
    finally:
        await _close(rl)


async def test_a_failed_claim_is_retried_well_inside_the_pending_lease(
    monkeypatch, redis_client, org_id, cleanup
):
    """Findings 3 and 4: a claim that hits a Redis error is retried after claim_retry_base_s (production
    interval, not heartbeat_renew_s), so the answered call is claimed before its pending lease dies."""
    from api.services.call_concurrency import service as service_module

    rl = RateLimiter()
    monkeypatch.setattr(service_module, "rate_limiter", rl)
    run_id = org_id
    cleanup.attempts.append("blip")
    cleanup.runs.append(run_id)
    real_claim = rl.claim_slot
    calls: list[float] = []

    async def flaky_claim(workflow_run_id):
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise RedisConnectionError("blip")
        return await real_claim(workflow_run_id)

    monkeypatch.setattr(rl, "claim_slot", flaky_claim)
    try:
        await _admit_and_bind(rl, org_id, "blip", run_id, 5)
        pending = await redis_client.zscore(keys.org_key(org_id), "blip")
        claimed = pending + rl.durations.slot_ttl - rl.durations.pending_ttl_s
        async with CallConcurrencyService().hold_run_slot(run_id):
            score = await _wait_for_score(
                redis_client, keys.org_key(org_id), "blip", claimed
            )
            assert score >= claimed, "the failed claim was not retried"
        assert len(calls) == 2
        assert calls[1] - calls[0] < rl.durations.pending_ttl_s
    finally:
        await _close(rl)


async def test_cancelled_while_claiming_still_releases(monkeypatch):
    """Finding 5: a CancelledError during the entry claim must not skip the release."""
    from api.services.call_concurrency import service as service_module

    entered = asyncio.Event()

    async def stuck_claim(workflow_run_id):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(service_module.rate_limiter, "claim_slot", stuck_claim)
    service = CallConcurrencyService()
    service.unregister_active_call = AsyncMock(return_value=False)

    async def call():
        async with service.hold_run_slot(77):
            pass

    task = asyncio.create_task(call())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    service.unregister_active_call.assert_awaited_once_with(77)


@pytest.mark.parametrize(
    ("carrier", "raw"),
    [
        ("telnyx", "timeout"),
        ("telnyx", "no_answer"),
        ("telnyx", "busy"),
        ("telnyx", "call_rejected"),
        ("vonage", "timeout"),
        ("vonage", "unanswered"),
        ("vonage", "cancelled"),
        ("vonage", "rejected"),
        ("vonage", "busy"),
        ("vonage", "failed"),
        ("plivo", "timeout"),
        ("plivo", "no-answer"),
        ("plivo", "busy"),
        ("plivo", "cancel"),
        ("twilio", "no-answer"),
        ("twilio", "busy"),
        ("twilio", "failed"),
        ("twilio", "canceled"),
    ],
)
def test_every_carrier_no_connect_cause_is_a_status_that_releases(carrier, raw):
    """Amendment (iii): each carrier's no-connect cause normalises to a status the status processor sends to the
    release funnel (the parametrized callback test above proves each status does)."""
    from api.enums import TelephonyCallStatus
    from api.services.telephony.providers.plivo.provider import PlivoProvider
    from api.services.telephony.providers.telnyx.provider import TelnyxProvider
    from api.services.telephony.providers.vonage.provider import VonageProvider
    from api.services.telephony.status_processor import (
        TERMINAL_NOT_CONNECTED_STATUSES,
    )

    if carrier == "telnyx":
        status = TelnyxProvider._resolve_status("call.hangup", {"hangup_cause": raw})
    elif carrier == "vonage":
        status = VonageProvider.parse_status_callback(None, {"status": raw})["status"]
    elif carrier == "plivo":
        status = PlivoProvider.parse_status_callback(None, {"CallStatus": raw})[
            "status"
        ]
    else:  # Twilio posts the normalised value itself
        status = raw
    assert TelephonyCallStatus.from_raw(status) in TERMINAL_NOT_CONNECTED_STATUSES


@pytest.mark.parametrize("status", ["initiated", "ringing"])
async def test_ringing_callback_restarts_the_pending_lease_of_an_outbound_call(
    monkeypatch, redis_client, fake_redis_clock, org_id, cleanup, status
):
    """Amendment (ii): a CPS queue before the ringing can outlast the lease sized at dial; the carrier's
    initiated/ringing callback restarts it. It never shortens a claimed slot and never revives an expired one."""
    from api.services.call_concurrency import service as service_module
    from api.services.telephony.status_processor import (
        StatusCallbackRequest,
        _process_status_update,
    )

    rl = RateLimiter(test_clock=fake_redis_clock.key)
    monkeypatch.setattr(service_module, "rate_limiter", rl)
    run_id = org_id
    cleanup.attempts.append("queued")
    cleanup.runs.append(run_id)
    lease = rl.durations.outbound_pending_ttl_s
    db = AsyncMock()
    db.get_workflow_run_by_id.return_value = SimpleNamespace(
        logs={}, campaign_id=None, state="initialized"
    )

    async def callback():
        with patch("api.services.telephony.status_processor.db_client", db):
            await _process_status_update(
                run_id, StatusCallbackRequest(call_id="c-1", status=status)
            )

    try:
        await _admit_and_bind(rl, org_id, "queued", run_id, 5, outbound=True)
        await fake_redis_clock.advance(lease - 10)  # still queued at the carrier
        await callback()
        await fake_redis_clock.advance(lease - 10)  # past the lease sized at dial
        assert await rl.get_concurrent_count(org_id) == 1
        for key in (keys.org_key(org_id), keys.fleet_key()):
            assert await redis_client.zscore(key, "queued") > fake_redis_clock.now

        assert await rl.claim_slot(run_id)
        claimed = await redis_client.zscore(keys.org_key(org_id), "queued")
        await callback()  # a late ringing callback after the answer
        assert await redis_client.zscore(keys.org_key(org_id), "queued") == claimed

        await rl.release_slot(org_id=org_id, attempt_id="queued")
        await _admit_and_bind(rl, org_id, "queued", run_id + 1, 5, outbound=True)
        cleanup.runs.append(run_id + 1)
        await fake_redis_clock.advance(lease + 1)  # expired before any callback
        await rl.extend_pending_slot(org_id=org_id, attempt_id="queued")
        assert await rl.get_concurrent_count(org_id) == 0
    finally:
        await _close(rl)
