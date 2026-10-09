"""Single source of call/slot/drain/grace durations (VOZ-AC-B3-55). Only the cell ceiling is written;
everything else derives from it and is asserted at startup of each role."""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

CEILING_ENV = "CELL_CALL_DURATION_CEILING_S"
DEFAULT_CEILING_S = 1200
_CEILING_HINT = f"set {CEILING_ENV} (cell.call_duration_ceiling_s) to a positive whole number of seconds"


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
        if ceiling_s < 1:
            raise DurationsError(
                "knob_out_of_range",
                f"call duration ceiling {ceiling_s}s is not positive",
                _CEILING_HINT,
            )
        d = cls(ceiling=ceiling_s, **margins)
        if not (d.ceiling < d.slot_ttl <= d.drain_max < d.grace):
            raise DurationsError(
                "durations_incoherent",
                f"invariant broken for ceiling={ceiling_s}",
                "check cell margins in the cell yaml",
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


def main(where: str) -> int:
    """Startup assertion of a role (scripts/xpand/require_db_head.sh): one line on stderr when the config is incoherent."""
    try:
        Durations.from_env()
    except DurationsError as e:
        print(
            f"code={e.code} where={where} reason={e.reason} hint={e.hint}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "durations"))
