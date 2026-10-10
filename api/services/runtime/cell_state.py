"""Runtime state of a cell worker that the health endpoint and the drain/heartbeat work read (VOZ-AC-B3-60/61/62).

One source for "is this worker draining" and "calls per pod". The API route and later service code import from here.
"""

import os
from functools import cache

DEFAULT_CALL_K_P = 4
# Must equal the default in scripts/xpand/call_entrypoint.sh, which writes the flag (a test pins both).
DEFAULT_DRAIN_FLAG_FILE = "/tmp/xpand_draining"


def parse_call_k_p(raw: str | None) -> int:
    """CALL_K_P as an int >= 1; unset means the default. Raises ValueError otherwise."""
    if raw is None or raw == "":
        return DEFAULT_CALL_K_P
    value = int(raw)
    if value < 1:
        raise ValueError(f"CALL_K_P must be >= 1, got {value}")
    return value


@cache
def _resolve_call_k_p(raw: str | None) -> int:
    # Cached per raw value so an invalid one warns once, not on every poll of the health endpoint.
    try:
        return parse_call_k_p(raw)
    except ValueError:
        # Imported here so the start gate (cell_startup) stays stdlib-only, like durations.py.
        from loguru import logger

        logger.warning(f"invalid CALL_K_P, using {DEFAULT_CALL_K_P}")
        return DEFAULT_CALL_K_P


def call_k_p() -> int:
    """Calls per pod. In a cell an invalid value never gets here: cell_startup refuses to start."""
    return _resolve_call_k_p(os.environ.get("CALL_K_P"))


def is_draining() -> bool:
    """True once the call entrypoint received SIGTERM (it creates the flag file and removes it at start)."""
    return os.path.exists(os.environ.get("DRAIN_FLAG_FILE", DEFAULT_DRAIN_FLAG_FILE))
