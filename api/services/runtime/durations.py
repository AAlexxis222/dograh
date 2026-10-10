"""Single source of call/slot/drain/grace durations (VOZ-AC-B3-55). Only the cell ceiling is written;
everything else derives from it and is asserted at startup of each role."""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

from api.enums import WorkflowRunMode

CEILING_ENV = "CELL_CALL_DURATION_CEILING_S"
STOP_GRACE_ENV = "CELL_STOP_GRACE_S"
DRAIN_TIMEOUT_ENV = "DRAIN_TIMEOUT"
DRAIN_MAX_WAIT_ENV = "DRAIN_MAX_WAIT"
DRAIN_INITIAL_DELAY_ENV = "DRAIN_INITIAL_DELAY"
DEFAULT_CEILING_S = 1200
# Default max_call_duration of a workflow; the cell ceiling cannot be lower (max_call_duration <= ceiling).
DEFAULT_MAX_CALL_DURATION_S = 300
_CEILING_HINT = (
    f"set {CEILING_ENV} (cell.call_duration_ceiling_s) to a whole number of seconds "
    f">= {DEFAULT_MAX_CALL_DURATION_S} (the default max_call_duration)"
)
_RERENDER_HINT = (
    "re-render the cell env from the ceiling: python -m scripts.xpand.render_durations"
)
# Deployed values that must not be below the ones derived from the ceiling.
_FLOOR_ENVS = (STOP_GRACE_ENV, DRAIN_MAX_WAIT_ENV, DRAIN_TIMEOUT_ENV)
# An outbound call must ring at least this long before it is abandoned (US TSR, 16 CFR 310.4(b)(4)).
MIN_RING_TIMEOUT_S = 15
# Ring timeout of an outbound call per carrier, keyed by WorkflowRunMode value: the same string is each provider's
# PROVIDER_NAME and the mode of every run it carries, so builders and status callbacks look up one key space. Each
# value is the carrier's own documented default, so no call rings longer or shorter than it did before. The builders
# that send a ring field send exactly this value
# (Twilio Timeout, Vonage ringing_timer, Telnyx timeout_secs, ARI originate timeout), which keeps the outbound
# pending lease (ring + pending_ttl_s) a true bound. Plivo: its API reference states no default for ring_timeout
# (https://plivo.com/docs/voice/api/calls), so its builder sends none and 120 here is UNVERIFIED, used only to size
# the lease. Carriers whose builder has no ring field (exotel, vobiz, cloudonix) size the lease on
# DEFAULT_RING_TIMEOUT_S and rely on the ringing re-extension (initiated/ringing callbacks) and, past it, on the late
# claim's slot_late_claim_overcommit event (accepted, VOZ-N0-21).
CARRIER_RING_TIMEOUT_S = {
    WorkflowRunMode.TWILIO.value: 60,
    WorkflowRunMode.VONAGE.value: 60,
    WorkflowRunMode.TELNYX.value: 30,
    WorkflowRunMode.ARI.value: 30,
    WorkflowRunMode.PLIVO.value: 120,
}
DEFAULT_RING_TIMEOUT_S = 60
# The claim retry cap is pending_ttl_s // 3: below 3 s it would be 0 and a failed claim would retry without pausing.
MIN_PENDING_TTL_S = 3
# Upper bound of the renewal period of a claimed slot (VOZ-AC-B3-66: heartbeat_renew_s <= slot_ttl / 3, default 300).
HEARTBEAT_RENEW_DEFAULT_S = 300


class DurationsError(ValueError):
    """VOZ-AC-B0-28 shape: stable ``code``, what is wrong (``reason``), where, and what to do (``hint``)."""

    def __init__(
        self, code: str, reason: str, hint: str, *, where: str = "cell durations"
    ):
        super().__init__(f"{code} at {where}: {reason} (hint: {hint})")
        self.code, self.reason, self.hint, self.where = code, reason, hint, where


def check_ring_floor(rings: Mapping[str, int]) -> None:
    """Every ring timeout respects the US TSR minimum ring (checked once, on the module's own table, at import)."""
    short = {carrier: s for carrier, s in rings.items() if s < MIN_RING_TIMEOUT_S}
    if short:
        raise DurationsError(
            "knob_out_of_range",
            f"ring timeout {short} is below the {MIN_RING_TIMEOUT_S}s minimum ring (US TSR, 16 CFR 310.4(b)(4))",
            f"keep every ring timeout >= {MIN_RING_TIMEOUT_S} (CARRIER_RING_TIMEOUT_S, "
            "api/services/runtime/durations.py)",
        )


check_ring_floor({**CARRIER_RING_TIMEOUT_S, "default": DEFAULT_RING_TIMEOUT_S})


def ring_timeout_for(carrier: str) -> int:
    """The ring timeout of an outbound call through ``carrier`` (a WorkflowRunMode value, CARRIER_RING_TIMEOUT_S)."""
    return CARRIER_RING_TIMEOUT_S.get(carrier, DEFAULT_RING_TIMEOUT_S)


@dataclass(frozen=True)
class Durations:
    ceiling: int
    slot_margin_s: int = 60
    drain_margin_s: int = 60
    pre_stop_delay_s: int = 15
    sigterm_headroom_s: int = 30
    # admission.pending_ttl_s: an admitted slot no worker has claimed yet expires after this (VOZ-AC-B3-22 lease).
    pending_ttl_s: int = 30
    # A worker whose claim failed retries after base, 2x base, ... up to claim_retry_cap_s.
    claim_retry_base_s: int = 1
    # Retry-After of an admission refused because the slot backend is down (HTTP 503, RFC 9110 §10.2.3).
    admission_retry_after_s: int = 5

    def __post_init__(self) -> None:
        # The invariants live in the type: no Durations exists that breaks them, however it was built.
        if self.ceiling < DEFAULT_MAX_CALL_DURATION_S:
            raise DurationsError(
                "knob_out_of_range",
                f"call duration ceiling {self.ceiling}s is below the default max_call_duration {DEFAULT_MAX_CALL_DURATION_S}s",
                _CEILING_HINT,
            )
        if self.pending_ttl_s < MIN_PENDING_TTL_S:
            raise DurationsError(
                "knob_out_of_range",
                f"pending_ttl_s {self.pending_ttl_s}s is below {MIN_PENDING_TTL_S}s: the claim retry cap "
                "(pending_ttl_s // 3) would be 0 and a failed claim would retry without pausing",
                f"keep pending_ttl_s >= {MIN_PENDING_TTL_S} (admission.pending_ttl_s, "
                "api/services/runtime/durations.py)",
            )
        if not (self.ceiling < self.slot_ttl <= self.drain_max < self.grace):
            raise DurationsError(
                "durations_incoherent",
                f"ceiling < slot_ttl <= drain_max < grace is broken for ceiling={self.ceiling}s with margins "
                f"slot={self.slot_margin_s}s, drain={self.drain_margin_s}s, pre-stop={self.pre_stop_delay_s}s, "
                f"SIGTERM headroom={self.sigterm_headroom_s}s",
                "the margins are constants of this module (api/services/runtime/durations.py); "
                f"keep them positive, or change the ceiling with {CEILING_ENV}",
            )

    @classmethod
    def from_cell(cls, ceiling_s: int, **margins: int) -> "Durations":
        return cls(ceiling=ceiling_s, **margins)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Durations":
        raw = (os.environ if environ is None else environ).get(CEILING_ENV, "").strip()
        if not raw:
            return cls.from_cell(DEFAULT_CEILING_S)
        try:
            ceiling = int(raw)
        except ValueError:
            raise DurationsError(
                "knob_invalid",
                f"{CEILING_ENV}={raw!r} is not a whole number of seconds",
                _CEILING_HINT,
            ) from None
        return cls.from_cell(ceiling)

    @property
    def slot_ttl(self) -> int:
        return self.ceiling + self.slot_margin_s

    @property
    def drain_max(self) -> int:
        return self.ceiling + self.drain_margin_s

    @property
    def grace(self) -> int:
        return self.pre_stop_delay_s + self.drain_max + self.sigterm_headroom_s

    @property
    def semaphore_ttl(self) -> int:
        return self.slot_ttl

    @property
    def heartbeat_renew_s(self) -> int:
        return min(HEARTBEAT_RENEW_DEFAULT_S, self.slot_ttl // 3)

    def outbound_pending_ttl_s(self, carrier: str) -> int:
        """Pending lease of an outbound admission through ``carrier`` (a WorkflowRunMode value): its ringing, then
        the inbound margin."""
        return ring_timeout_for(carrier) + self.pending_ttl_s

    @property
    def claim_retry_cap_s(self) -> int:
        """Retries stay well inside the pending lease, so a failed claim is retried before the slot expires."""
        return self.pending_ttl_s // 3

    def check_workflow_max(self, max_call_duration_s: int) -> None:
        if max_call_duration_s > self.ceiling:
            raise DurationsError(
                "knob_out_of_range",
                f"max_call_duration {max_call_duration_s}s > cell ceiling {self.ceiling}s",
                "raise cell.call_duration_ceiling_s via the cell runbook (drained restart)",
            )

    def deployed_env(self) -> dict[str, int]:
        """Every env variable of a cell that derives from the ceiling, with its derived value: what the render
        writes and what check_deployed compares against."""
        return {
            CEILING_ENV: self.ceiling,
            STOP_GRACE_ENV: self.grace,
            DRAIN_TIMEOUT_ENV: self.drain_max,
            DRAIN_MAX_WAIT_ENV: self.drain_max,
            DRAIN_INITIAL_DELAY_ENV: self.pre_stop_delay_s,
        }

    def check_deployed(self, environ: Mapping[str, str]) -> None:
        """The rendered numbers a role was started with must not undercut the ones derived from its ceiling.

        Unset = not checked. Smaller is the dangerous side: a grace or a drain shorter than the longest call cuts it,
        and so does an initial delay that eats the drain window out of the grace.
        """
        derived = self.deployed_env()
        del derived[CEILING_ENV]  # the source itself: from_env reads it
        found: dict[str, int] = {}
        for name, expected in derived.items():
            raw = environ.get(name, "").strip()
            if not raw:
                continue
            try:
                found[name] = int(raw)
            except ValueError:
                raise DurationsError(
                    "knob_invalid",
                    f"{name}={raw!r} is not a whole number of seconds",
                    _RERENDER_HINT,
                ) from None
            if name in _FLOOR_ENVS and found[name] < expected:
                raise DurationsError(
                    "durations_incoherent",
                    f"{name}={found[name]}s is below the {expected}s derived from {CEILING_ENV}={self.ceiling}",
                    "the ceiling was changed without re-rendering the cell env: "
                    "python -m scripts.xpand.render_durations, then a drained restart",
                )
        budget = (DRAIN_INITIAL_DELAY_ENV, DRAIN_MAX_WAIT_ENV, STOP_GRACE_ENV)
        if all(name in found for name in budget):
            needed = (
                found[DRAIN_INITIAL_DELAY_ENV]
                + found[DRAIN_MAX_WAIT_ENV]
                + self.sigterm_headroom_s
            )
            if needed > found[STOP_GRACE_ENV]:
                raise DurationsError(
                    "durations_incoherent",
                    f"{DRAIN_INITIAL_DELAY_ENV}={found[DRAIN_INITIAL_DELAY_ENV]}s + {DRAIN_MAX_WAIT_ENV}="
                    f"{found[DRAIN_MAX_WAIT_ENV]}s + {self.sigterm_headroom_s}s SIGTERM headroom = {needed}s "
                    f"exceeds {STOP_GRACE_ENV}={found[STOP_GRACE_ENV]}s: the orchestrator's KILL lands mid-drain",
                    _RERENDER_HINT,
                )


@cache
def cell_durations() -> Durations:
    """The durations of this process, resolved once from the environment: every consumer (the workflow schema
    bound, the validator, the rate limiter) reads this one instance, so they cannot see different ceilings."""
    return Durations.from_env()


def main(where: str) -> int:
    """Startup assertion of a role (scripts/xpand/require_db_head.sh): one line on stderr when the config is incoherent."""
    try:
        Durations.from_env().check_deployed(os.environ)
    except DurationsError as e:
        print(
            f"code={e.code} where={where} reason={e.reason} hint={e.hint}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "durations"))
