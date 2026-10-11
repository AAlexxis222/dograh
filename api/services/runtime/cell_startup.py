"""Startup assertion of a cell role (VOZ-AC-B3-61, VOZ-AC-B6-52, VOZ-AC-B6-16), and of the credential master key in every role. Fail closed, VOZ-AC-B0-28 output.

Light on purpose (no FastAPI import): the api lifespan calls it, and scripts/xpand/require_db_head.sh runs it as
`python -m api.services.runtime.cell_startup <role>` for every role, so arq and coordinators are covered too.
"""

import os
import sys

from api.services.runtime.cell_state import parse_call_k_p
from api.services.security import credential_box

# Levels at or above INFO. Anything else (DEBUG, TRACE, numbers, unknown names, unset) can log caller text.
_SAFE_LOG_LEVELS = {"INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}


class CellStartupError(RuntimeError):
    """VOZ-AC-B0-28 shape: stable ``code``, ``where``, non-empty ``reason`` and ``hint``."""

    def __init__(self, code: str, where: str, reason: str, hint: str):
        super().__init__(f"code={code} where={where} reason={reason} hint={hint}")


def _assert_credentials_master_key(where: str, *, in_cell: bool) -> None:
    """VOZ-AC-B6-16. A key that is set must be valid in every role, cell or not: a malformed one would otherwise
    surface as a generic error at the first credential write. Outside a cell a missing key is allowed and said out
    loud (VOZ-N0-24c: credentials are then stored in clear); in a cell it stops the role. Validated uncached, so the
    check reads the environment as it is now, not a key cached earlier in this process."""
    if not in_cell and not credential_box.master_key_items(
        os.environ.get(credential_box.ENV, "")
    ):
        # stderr, not loguru: this module stays light (require_db_head.sh runs it with any python on PATH).
        print(
            f"code=credentials_at_rest_plaintext where={where} reason={credential_box.ENV} is not set, so "
            f"credentials are stored in clear hint=set {credential_box.ENV} (key_id=base64 of 32 bytes) and run "
            "alembic upgrade head",
            file=sys.stderr,
        )
        return
    try:
        credential_box.validate_master_key()
    except credential_box.MasterKeyInvalid as e:
        raise CellStartupError(
            e.code, where, f"{e.where}: {e.reason}", e.hint
        ) from None


def assert_cell_startup_config(where: str = "api.app") -> None:
    """Fail closed in a cell (CELL_ROLE set). Outside a cell only the credential master key is checked."""
    if not os.environ.get("CELL_ROLE"):
        _assert_credentials_master_key(where, in_cell=False)
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
    _assert_credentials_master_key(where, in_cell=True)
    # Durations are not checked here: the gate in scripts/xpand/require_db_head.sh owns them for every role.


def main(where: str) -> int:
    try:
        assert_cell_startup_config(where)
    except CellStartupError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "cell_startup"))
