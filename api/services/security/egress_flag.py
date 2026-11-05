"""The URL guard flag (VOZ-AC-B5-91): its name and its rule, in one place.

Standard library only: the cell start check (api/services/runtime/cell_startup.py) imports it and must stay light.
"""

import os

# The guard has its own flag, so a cell (DEPLOYMENT_MODE=oss) can enforce it without changing the other branches
# that mode governs.
URL_GUARD_FLAG = "URL_GUARD_ENFORCE"


def url_guard_enforced() -> bool:
    """Read at call time; only the exact value "true" enforces, anything else is the guard off."""
    return os.environ.get(URL_GUARD_FLAG) == "true"
