"""Startup assertion of a cell role (VOZ-AC-B3-61, VOZ-AC-B6-52). Fail closed, VOZ-AC-B0-28 output.

Light on purpose (no FastAPI import): the api lifespan calls it, and scripts/xpand/require_db_head.sh runs it as
`python -m api.services.runtime.cell_startup <role>` for every role, so arq and coordinators are covered too.
"""

import os
import sys

from api.services.runtime.durations import Durations, DurationsError

DEFAULT_CALL_K_P = 4
# Levels at or above INFO. Anything else (DEBUG, TRACE, numbers, unknown names, unset) can log caller text.
_SAFE_LOG_LEVELS = {"INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}


class CellStartupError(RuntimeError):
    """VOZ-AC-B0-28 shape: stable ``code``, ``where``, non-empty ``reason`` and ``hint``."""

    def __init__(self, code: str, where: str, reason: str, hint: str):
        super().__init__(f"code={code} where={where} reason={reason} hint={hint}")


def parse_call_k_p(raw: str | None) -> int:
    """CALL_K_P as an int >= 1; unset means the default. Raises ValueError otherwise."""
    if raw is None or raw == "":
        return DEFAULT_CALL_K_P
    value = int(raw)
    if value < 1:
        raise ValueError(f"CALL_K_P must be >= 1, got {value}")
    return value


def assert_cell_startup_config(where: str = "api.app") -> None:
    """Fail closed in a cell (CELL_ROLE set); a no-op anywhere else."""
    if not os.environ.get("CELL_ROLE"):
        return
    if not os.environ.get("DOGRAH_DEVOPS_SECRET"):
        raise CellStartupError(
            "devops_secret_missing",
            where,
            "DOGRAH_DEVOPS_SECRET is empty, so /api/v1/health/active-calls answers 503 and a deploy cannot drain",
            "set DOGRAH_DEVOPS_SECRET from the secrets store in the cell env",
        )
    # Unset is DEBUG (api/constants.py defaults it so).
    if os.environ.get("LOG_LEVEL", "DEBUG").upper() not in _SAFE_LOG_LEVELS:
        raise CellStartupError(
            "log_level_debug_in_cell",
            where,
            "LOG_LEVEL is below INFO or unset (unset means DEBUG), which logs caller text and phone numbers",
            "set LOG_LEVEL=INFO in the cell env",
        )
    try:
        parse_call_k_p(os.environ.get("CALL_K_P"))
    except ValueError as e:
        raise CellStartupError(
            "call_k_p_invalid",
            where,
            f"CALL_K_P is not an integer >= 1 ({e})",
            "set CALL_K_P to a positive integer, or unset it for the default of 4",
        ) from e
    try:
        Durations.from_env().check_deployed(os.environ)
    except DurationsError as e:
        raise CellStartupError(e.code, where, e.reason, e.hint) from e


def main(where: str) -> int:
    try:
        assert_cell_startup_config(where)
    except CellStartupError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "cell_startup"))
