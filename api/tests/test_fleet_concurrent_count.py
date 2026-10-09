"""Unit tests for RateLimiter.get_fleet_concurrent_count — the fleet-wide
autoscaling signal.

The count is a single read over the fleet zset (FLEET_CONCURRENT_KEY) that the
acquire/renew/release paths maintain; each member's score is its expiry in
Redis time (keys.py). These tests seed that one key in a real Redis and read it
with a test clock far in the future, so every member other tests left in the
shared key reads as expired and only the seeded ones count. The write side —
acquire mirrors a member in, release removes it, scoped slots never
double-count — runs in test_call_concurrency.py and test_slots_redis_time.py.

The behaviors that matter:
  - unexpired slots are counted; expired ones (score <= Redis TIME) are
    excluded without being deleted, so an orphaned call can't keep the metric
    high and block scale-down;
  - an empty fleet reads 0;
  - Redis errors propagate (the endpoint 503s) instead of reading as an idle
    fleet, which would be a scale-to-minimum instruction.
"""

import os
import uuid
from unittest.mock import AsyncMock

import pytest

from api.services.call_concurrency.rate_limiter import (
    FLEET_CONCURRENT_KEY,
    RateLimiter,
)

requires_redis = pytest.mark.skipif(
    "REDIS_URL" not in os.environ,
    reason="docker_missing: needs a real Redis (REDIS_URL via .env.test)",
)

_FAR_FUTURE_S = 10 * 365 * 86400


@pytest.fixture
async def future_clock(fake_redis_clock):
    await fake_redis_clock.advance(_FAR_FUTURE_S)
    return fake_redis_clock


@requires_redis
@pytest.mark.asyncio
async def test_counts_unexpired_slots_and_excludes_expired_ones_without_writing(
    redis_client, future_clock
):
    now = future_clock.now
    tag = uuid.uuid4().hex
    members = {f"fresh-{tag}-{i}": now + 100 for i in range(2)}
    members |= {f"expired-{tag}-{i}": now - 5000 for i in range(3)}
    members[f"expires-now-{tag}"] = now
    await redis_client.zadd(FLEET_CONCURRENT_KEY, members)
    rl = RateLimiter(test_clock=future_clock.key)
    try:
        assert await rl.get_fleet_concurrent_count() == 2
        assert await redis_client.zmscore(FLEET_CONCURRENT_KEY, list(members)) == [
            pytest.approx(score) for score in members.values()
        ]  # read-only: nothing was purged
    finally:
        await redis_client.zrem(FLEET_CONCURRENT_KEY, *members)
        await rl.close()


@requires_redis
@pytest.mark.asyncio
async def test_empty_fleet_is_zero(future_clock):
    rl = RateLimiter(test_clock=future_clock.key)
    try:
        assert await rl.get_fleet_concurrent_count() == 0
    finally:
        await rl.close()


@pytest.mark.asyncio
async def test_redis_error_propagates():
    class _Boom:
        async def eval(self, *args):
            raise ConnectionError("redis down")

    rl = RateLimiter()
    rl._get_redis = AsyncMock(return_value=_Boom())  # type: ignore[method-assign]
    # A failed read must NOT report 0 (an idle fleet scales to minimum); it
    # propagates so the autoscale-metric endpoint can respond 503 and KEDA's
    # HPA holds the current replica count.
    with pytest.raises(ConnectionError):
        await rl.get_fleet_concurrent_count()
