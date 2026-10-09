"""Single source of call/slot/drain/grace durations (VOZ-AC-B3-55). Only the cell ceiling is written;
everything else derives from it and is asserted at startup of each role."""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

CEILING_ENV = "CELL_CALL_DURATION_CEILING_S"
STOP_GRACE_ENV = "CELL_STOP_GRACE_S"
DEFAULT_CEILING_S = 1200
# Default max_call_duration of a workflow; the cell ceiling cannot be lower (max_call_duration <= ceiling).
DEFAULT_MAX_CALL_DURATION_S = 300
_CEILING_HINT = (
    f"set {CEILING_ENV} (cell.call_duration_ceiling_s) to a whole number of seconds "
    f">= {DEFAULT_MAX_CALL_DURATION_S} (the default max_call_duration)"
)


class DurationsError(ValueError):
    """VOZ-AC-B0-28 shape: stable ``code``, what is wrong (``reason``), where, and what to do (``hint``)."""

    def __init__(
        self, code: str, reason: str, hint: str, *, where: str = "cell durations"
    ):
        super().__init__(f"{code} at {where}: {reason} (hint: {hint})")
        self.code, self.reason, self.hint, self.where = code, reason, hint, where


@dataclass(frozen=True)
class Durations:
    ceiling: int
    slot_margin_s: int = 60
    drain_margin_s: int = 60
    pre_stop_delay_s: int = 15
    sigterm_headroom_s: int = 30

    @classmethod
    def from_cell(cls, ceiling_s: int, **margins: int) -> "Durations":
        if ceiling_s < DEFAULT_MAX_CALL_DURATION_S:
            raise DurationsError(
                "knob_out_of_range",
                f"call duration ceiling {ceiling_s}s is below the default max_call_duration {DEFAULT_MAX_CALL_DURATION_S}s",
                _CEILING_HINT,
            )
        d = cls(ceiling=ceiling_s, **margins)
        if not (d.ceiling < d.slot_ttl <= d.drain_max < d.grace):
            raise DurationsError(
                "durations_incoherent",
                f"ceiling < slot_ttl <= drain_max < grace is broken for ceiling={ceiling_s}s with margins {margins or 'at their defaults'}",
                "the margins are constants of this module (api/services/runtime/durations.py); "
                f"keep them positive, or change the ceiling with {CEILING_ENV}",
            )
        return d

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
        return self.slot_ttl // 3

    def check_workflow_max(self, max_call_duration_s: int) -> None:
        if max_call_duration_s > self.ceiling:
            raise DurationsError(
                "knob_out_of_range",
                f"max_call_duration {max_call_duration_s}s > cell ceiling {self.ceiling}s",
                "raise cell.call_duration_ceiling_s via the cell runbook (drained restart)",
            )

    def check_deployed(self, environ: Mapping[str, str]) -> None:
        """The rendered numbers a role was started with must not undercut the ones derived from its ceiling.

        Unset = not checked. Smaller is the dangerous side: a grace or a drain shorter than the longest call cuts it.
        """
        for name, derived in (
            (STOP_GRACE_ENV, self.grace),
            ("DRAIN_MAX_WAIT", self.drain_max),
            ("DRAIN_TIMEOUT", self.drain_max),
        ):
            raw = environ.get(name, "").strip()
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError:
                raise DurationsError(
                    "knob_invalid",
                    f"{name}={raw!r} is not a whole number of seconds",
                    "re-render the cell env from the ceiling: python -m scripts.xpand.render_durations",
                ) from None
            if value < derived:
                raise DurationsError(
                    "durations_incoherent",
                    f"{name}={value}s is below the {derived}s derived from {CEILING_ENV}={self.ceiling}",
                    "the ceiling was changed without re-rendering the cell env: "
                    "python -m scripts.xpand.render_durations, then a drained restart",
                )


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
