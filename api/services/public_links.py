"""Policy of the public download links of a call's recording and transcript (VOZ-N0-30).

Pure on purpose: no database, so the db clients and the routes share one definition. A link lives
``PUBLIC_LINK_TTL_H`` hours from the moment it is minted (not from the call), can be revoked for good, and is off
inside a cell unless the organization sets ``public_links_enabled: true`` in its workflow configuration defaults.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from api.services.runtime.cell_startup import is_cell
from api.services.runtime.durations import cell_durations

# Key of the organization document (WORKFLOW_CONFIGURATION_DEFAULTS) that turns links on inside a cell.
ORG_KEY = "public_links_enabled"


def now() -> datetime:
    """The one clock of link minting and checking, so a test moves it in one place."""
    return datetime.now(UTC)


def links_enabled(organization_defaults: Any) -> bool:
    """Outside a cell links stay on, as before. Inside one they are off unless the organization document says
    exactly ``true`` (a string, a 1 or an unreadable document is off: fail closed)."""
    if not is_cell():
        return True
    return (
        isinstance(organization_defaults, dict)
        and organization_defaults.get(ORG_KEY) is True
    )


def expiry_from(minted_at: datetime) -> datetime:
    return minted_at + timedelta(hours=cell_durations().public_link_ttl_h)


def is_expired(expires_at: datetime | None, at: datetime) -> bool:
    """``NULL`` is expired: a row minted before the column existed has no deadline, so it never opens."""
    return expires_at is None or at >= expires_at
