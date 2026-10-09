import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from loguru import logger

from api.constants import DEFAULT_ORG_CONCURRENCY_LIMIT
from api.db import db_client
from api.enums import OrganizationConfigurationKey, PostHogEvent
from api.services.call_concurrency import keys
from api.services.call_concurrency.rate_limiter import (
    AdmissionBackendUnavailable,
    SlotLease,
    rate_limiter,
)
from api.services.posthog_client import capture_event


@dataclass(frozen=True)
class CallConcurrencySlot:
    organization_id: int
    slot_id: str
    max_concurrent: int
    source: str
    scope_key: str | None = None
    scope_max_concurrent: int | None = None


class CallConcurrencyLimitError(Exception):
    """Raised when admission refuses a call: no free slot (``reason`` "concurrent_call_limit") or, failing
    closed, no answer from the slot backend (``reason`` "admission_backend_unavailable", VOZ-AC-B3-65-bis)."""

    def __init__(
        self,
        *,
        organization_id: int,
        source: str,
        wait_time: float,
        max_concurrent: int,
        reason: str = "concurrent_call_limit",
        what: str | None = None,
        where: str | None = None,
        hint: str | None = None,
    ):
        self.organization_id = organization_id
        self.source = source
        self.wait_time = wait_time
        self.max_concurrent = max_concurrent
        # ``reason`` is the stable code; ``what`` / ``where`` / ``hint`` complete the VOZ-AC-B0-28 record.
        self.reason = reason
        self.what = what
        self.where = where
        self.hint = hint
        # Callers answer this case differently (carrier code, HTTP 503): the org is not full, Redis is down.
        self.backend_unavailable = reason == AdmissionBackendUnavailable.reason
        super().__init__(
            f"Call admission refused for org {organization_id}: {reason} "
            f"(source={source}, limit={max_concurrent}, waited={wait_time:.1f}s)"
            + (f" (hint: {hint})" if hint else "")
        )

    # One place maps the refusal's true reason to each channel's answer (VOZ-AC-B3-65-bis); the audible wording
    # of the carrier answer is VOZ-N0-29's.
    def failure(self) -> dict:
        """The VOZ-AC-B0-28 record of the refusal: stable code, what happened, where, and what to do."""
        return {
            "code": self.reason,
            "reason": self.what,
            "where": self.where,
            "hint": self.hint,
        }

    def http_answer(self) -> dict:
        """HTTPException kwargs: 429 for a full org, 503 with Retry-After (RFC 9110 §10.2.3) when Redis is down."""
        if self.backend_unavailable:
            retry_after = rate_limiter.durations.admission_retry_after_s
            return {
                "status_code": 503,
                "detail": self.failure(),
                "headers": {"Retry-After": str(retry_after)},
            }
        return {"status_code": 429, "detail": "Concurrent call limit reached"}

    def ws_close(self) -> dict:
        """WebSocket close kwargs: 1008 for a full org, 1013 (try again later) when Redis is down. A close reason
        holds at most 123 bytes (RFC 6455 §5.5), so it carries code, where and hint, not the whole record."""
        if self.backend_unavailable:
            return {
                "code": 1013,
                "reason": f"{self.reason} at {self.where}: {self.hint}",
            }
        return {"code": 1008, "reason": "Concurrent call limit reached"}

    def client_error(self) -> dict:
        """Error payload sent to a browser client over the signaling socket."""
        if self.backend_unavailable:
            return {
                "error_type": self.reason,
                "message": "Service temporarily unavailable",
                **self.failure(),
            }
        return {
            "error_type": "concurrency_limit_exceeded",
            "message": "Concurrent call limit reached",
        }


class WorkflowRunSlotAlreadyBoundError(Exception):
    """Raised when a workflow run already owns a concurrent call slot."""

    def __init__(self, workflow_run_id: int):
        self.workflow_run_id = workflow_run_id
        super().__init__(
            f"Workflow run {workflow_run_id} already has an active call slot"
        )


class CallConcurrencyService:
    def __init__(self):
        self.default_concurrent_limit = int(DEFAULT_ORG_CONCURRENCY_LIMIT)

    async def get_org_concurrent_limit(self, organization_id: int) -> int:
        """Get the concurrent call limit for an organization."""
        try:
            config = await db_client.get_configuration(
                organization_id,
                OrganizationConfigurationKey.CONCURRENT_CALL_LIMIT.value,
            )
            if config and config.value:
                value = config.value.get("value")
                if value is not None:
                    return int(value)
        except Exception as e:
            logger.warning(
                f"Error getting concurrent limit for org {organization_id}: {e}"
            )
        return self.default_concurrent_limit

    async def get_fleet_active_calls(self) -> int:
        """Total active calls across every org — the fleet-wide autoscaling
        signal scraped by /health/autoscale-metric. Thin passthrough so routes
        stay behind this facade; the Redis projection lives with the key schema
        in rate_limiter. Redis errors propagate (they must not read as an idle
        fleet — see get_fleet_concurrent_count)."""
        return await rate_limiter.get_fleet_concurrent_count()

    async def get_org_active_calls(self, organization_id: int) -> int:
        """Occupied org slots across workers, with storage failures propagated.

        Includes reservations made before a voice pipeline starts. Uses the
        same stale-slot expiry as concurrency enforcement.
        """
        return await rate_limiter.get_concurrent_count(
            organization_id, raise_on_error=True
        )

    async def acquire_org_slot(
        self,
        organization_id: int,
        *,
        source: str,
        timeout: float = 0,
        scope_key: str | None = None,
        scope_max_concurrent: int | None = None,
        retry_interval: float = 1,
        outbound_carrier: str | None = None,
    ) -> CallConcurrencySlot:
        """Acquire a slot in the org-wide concurrency counter.

        ``scope_key``/``scope_max_concurrent`` additionally bound a secondary
        counter (e.g. ``campaign:<id>``) so a source can cap its own
        concurrency without measuring — or being starved by — unrelated calls
        in the same org. An outbound call through ``outbound_carrier`` gets the
        longer pending lease that covers that carrier's ringing
        (durations.outbound_pending_ttl_s).
        """
        max_concurrent = await self.get_org_concurrent_limit(organization_id)
        if scope_max_concurrent is not None:
            scope_max_concurrent = int(scope_max_concurrent)

        attempt_id = keys.new_attempt_id()
        wait_start = time.time()
        while True:
            try:
                acquisition = await rate_limiter.acquire_slot(
                    org_id=organization_id,
                    attempt_id=attempt_id,
                    max_concurrent=max_concurrent,
                    scope_key=scope_key,
                    scope_max_concurrent=scope_max_concurrent,
                    outbound_carrier=outbound_carrier,
                )
            except AdmissionBackendUnavailable as e:
                logger.error(
                    f"Call admission failing closed for org {organization_id}: "
                    f"source={source}, {e}"
                )
                raise CallConcurrencyLimitError(
                    organization_id=organization_id,
                    source=source,
                    wait_time=time.time() - wait_start,
                    max_concurrent=max_concurrent,
                    reason=e.reason,
                    what=e.what,
                    where=e.where,
                    hint=e.hint,
                ) from e
            if acquisition:
                logger.info(
                    f"Acquired concurrent call slot for org {organization_id}: "
                    f"source={source}, active_calls="
                    f"{acquisition.active_count}/{max_concurrent}, "
                    f"slot_id={acquisition.slot_id}"
                )
                return CallConcurrencySlot(
                    organization_id=organization_id,
                    slot_id=acquisition.slot_id,
                    max_concurrent=max_concurrent,
                    source=source,
                    scope_key=scope_key,
                    scope_max_concurrent=scope_max_concurrent,
                )

            wait_time = time.time() - wait_start
            if wait_time >= timeout:
                current_count = await rate_limiter.get_concurrent_count(organization_id)
                scope_note = (
                    f", scope={scope_key} (limit={scope_max_concurrent})"
                    if scope_key
                    else ""
                )
                logger.warning(
                    f"Concurrent call limit reached for org {organization_id}: "
                    f"source={source}, active_calls={current_count}/{max_concurrent}"
                    f"{scope_note}, waited={wait_time:.1f}s"
                )
                properties = {
                    "event_source": "dograh",
                    "organization_id": organization_id,
                    "source": source,
                    "max_concurrent": max_concurrent,
                    "active_calls": current_count,
                    "waited_seconds": round(wait_time, 1),
                }
                if scope_key:
                    properties["scope_key"] = scope_key
                    properties["scope_max_concurrent"] = scope_max_concurrent
                await self._notify_limit_reached(organization_id, properties)
                raise CallConcurrencyLimitError(
                    organization_id=organization_id,
                    source=source,
                    wait_time=wait_time,
                    max_concurrent=max_concurrent,
                )

            logger.debug(
                f"Waiting for concurrent call slot for org {organization_id}, "
                f"source={source}, waited {wait_time:.1f}s"
            )
            await asyncio.sleep(min(retry_interval, max(0, timeout - wait_time)))

    async def _notify_limit_reached(
        self, organization_id: int, properties: dict
    ) -> None:
        """Fan the usage event out to every org member's provider_id, matching
        how MPS emits org-scoped billing events (billing_posthog_service.py)
        into the shared PostHog project. Never raises.

        NOTE: intentionally NOT attaching ``$groups`` (organization) to this
        event. PostHog evaluates a $groups event at both person and group
        scope, which double-triggers person-scoped workflows enrolled on the
        event. The org is still available as the ``organization_id`` property.
        """
        try:
            members = await db_client.get_organization_users(organization_id)
            if not members:
                logger.debug(
                    f"No users found for org {organization_id}; skipping "
                    "concurrent-call-limit PostHog event"
                )
                return
            for member in members:
                capture_event(
                    distinct_id=str(member.provider_id),
                    event=PostHogEvent.USAGE_CONCURRENT_CALL_LIMIT_REACHED,
                    properties=properties,
                )
        except Exception:
            logger.exception(
                "Failed to send concurrent-call-limit PostHog event for org "
                f"{organization_id}"
            )

    async def bind_workflow_run(
        self, slot: CallConcurrencySlot, workflow_run_id: int
    ) -> None:
        stored = await rate_limiter.store_workflow_slot_mapping_if_absent(
            workflow_run_id,
            slot.organization_id,
            slot.slot_id,
            scope_key=slot.scope_key,
            max_concurrent=slot.max_concurrent,
            scope_max_concurrent=slot.scope_max_concurrent,
        )
        if stored:
            return

        await self.release_slot(slot)
        raise WorkflowRunSlotAlreadyBoundError(workflow_run_id)

    async def register_active_call(
        self,
        organization_id: int,
        workflow_run_id: int,
        *,
        source: str,
        timeout: float = 0,
        scope_key: str | None = None,
        scope_max_concurrent: int | None = None,
        retry_interval: float = 1,
    ) -> CallConcurrencySlot:
        slot = await self.acquire_org_slot(
            organization_id,
            source=source,
            timeout=timeout,
            scope_key=scope_key,
            scope_max_concurrent=scope_max_concurrent,
            retry_interval=retry_interval,
        )
        await self.bind_workflow_run(slot, workflow_run_id)
        return slot

    async def unregister_active_call(self, workflow_run_id: int) -> bool:
        """Release the run's slot without ever raising.

        Callers invoke this from ``finally`` blocks during pipeline/socket
        teardown; a cleanup failure must not mask the original exception.
        The slot mapping survives a failed release, so a later cleanup path
        (status callback, StasisEnd) or the Redis stale timeout recovers it.
        """
        try:
            return await self.release_workflow_run_slot(workflow_run_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                f"Failed to release concurrent call slot for workflow run "
                f"{workflow_run_id}: {e}"
            )
            return False

    async def release_slot(self, slot: CallConcurrencySlot | None) -> bool:
        if slot is None:
            return False
        released = await rate_limiter.release_slot(
            org_id=slot.organization_id,
            attempt_id=slot.slot_id,
            scope_key=slot.scope_key,
        )
        return bool(released)

    async def release_workflow_run_slot(self, workflow_run_id: int) -> bool:
        """Release the org/campaign slot held by a workflow run, and its
        mapping, in one call to the release funnel. Every exit point of a run
        (worker end, carrier terminal callback, pre-pipeline failures, ARI
        teardown) lands here."""
        mapping = await rate_limiter.get_workflow_slot_mapping(workflow_run_id)
        if not mapping:
            return False

        org_id, slot_id, scope_key = mapping
        released = await rate_limiter.release_slot(
            org_id=org_id,
            attempt_id=slot_id,
            scope_key=scope_key,
            workflow_run_id=workflow_run_id,
        )
        if released is None:
            # Redis error while releasing: the script did nothing, so the
            # mapping is still there for a later cleanup path to retry
            # instead of orphaning a live slot until it expires.
            logger.warning(
                f"Failed to release concurrent slot for workflow run "
                f"{workflow_run_id}; keeping mapping for retry"
            )
            return False
        if released:
            logger.info(f"Released concurrent slot for workflow run {workflow_run_id}")
        else:
            logger.info(
                f"Concurrent slot mapping for workflow run {workflow_run_id} "
                "had no live slot; deleted stale mapping"
            )
        return released

    async def extend_ringing_slot(self, workflow_run_id: int, carrier: str) -> None:
        """``carrier``'s initiated/ringing callback: the outbound call is still on its way to an answer, so its
        pending lease restarts (a CPS queue can outlast the lease sized at dial). Never raises: the late claim
        on answer still re-checks the limit if this is missed."""
        try:
            mapping = await rate_limiter.get_workflow_slot_mapping(workflow_run_id)
            if mapping:
                org_id, slot_id, scope_key = mapping
                await rate_limiter.extend_pending_slot(
                    org_id=org_id,
                    attempt_id=slot_id,
                    carrier=carrier,
                    scope_key=scope_key,
                )
        except Exception as e:
            logger.warning(
                f"Pending slot extension failed for workflow run {workflow_run_id}: {e}"
            )

    @asynccontextmanager
    async def hold_run_slot(
        self, workflow_run_id: int, *, renew_interval_s: float | None = None
    ) -> AsyncIterator[None]:
        """Hold the run's slot for the life of the call on this worker.

        Entering claims the pending slot (phase 2 of the lease); a task of
        this call renews it every ``renew_interval_s`` (default
        ``heartbeat_renew_s``), independent of any worker heartbeat
        (VOZ-AC-B3-66); leaving cancels the renewal and releases through the
        funnel, by the claimed lease when there is one. Never raises on Redis
        errors: a failed claim is retried on a short bounded backoff.
        """
        hold = _SlotHold()
        renewal = None
        try:
            claimed = await self._claim(workflow_run_id, hold)
            if hold.lease is not None or not claimed:
                renewal = asyncio.create_task(
                    self._keep_run_slot(
                        workflow_run_id,
                        hold,
                        renew_interval_s or rate_limiter.durations.heartbeat_renew_s,
                    )
                )
            yield
        finally:
            try:
                if renewal is not None:
                    renewal.cancel()
                    await asyncio.wait([renewal])
            finally:
                await self._release_held(workflow_run_id, hold.lease)

    async def _claim(self, workflow_run_id: int, hold: "_SlotHold") -> bool:
        """True when Redis answered (``hold.lease`` is None if the run holds no slot), False on an error."""
        try:
            hold.lease = await rate_limiter.claim_slot(workflow_run_id)
            return True
        except Exception as e:
            logger.warning(f"Slot claim failed for workflow run {workflow_run_id}: {e}")
            return False

    async def _keep_run_slot(
        self, workflow_run_id: int, hold: "_SlotHold", interval: float
    ) -> None:
        durations = rate_limiter.durations
        retry = durations.claim_retry_base_s
        while hold.lease is None:
            # The claim failed: retry well inside the pending lease, not one renewal interval later.
            await asyncio.sleep(min(retry, interval))
            retry = min(retry * 2, durations.claim_retry_cap_s)
            if await self._claim(workflow_run_id, hold) and hold.lease is None:
                return  # released meanwhile: nothing left to hold
        lease = hold.lease
        while True:
            await asyncio.sleep(interval)
            try:
                await rate_limiter.renew_slot(
                    org_id=lease.org_id,
                    attempt_id=lease.attempt_id,
                    scope_key=lease.scope_key,
                    workflow_run_id=workflow_run_id,
                )
            except Exception as e:
                logger.warning(
                    f"Slot renewal failed for workflow run {workflow_run_id}: {e}"
                )

    async def _release_held(
        self, workflow_run_id: int, lease: SlotLease | None
    ) -> None:
        if lease is None:
            await self.unregister_active_call(workflow_run_id)
            return
        # By the lease, so the release does not depend on the mapping still being there.
        await rate_limiter.release_slot(
            org_id=lease.org_id,
            attempt_id=lease.attempt_id,
            scope_key=lease.scope_key,
            workflow_run_id=workflow_run_id,
        )


@dataclass
class _SlotHold:
    """The lease one call holds, shared by its claim, its renewal task and its release."""

    lease: SlotLease | None = None


call_concurrency = CallConcurrencyService()
