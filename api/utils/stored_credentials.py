"""Reading a stored credentials document.

A leaf module on purpose: the db layer and the provider registry both need this rule,
and the db layer cannot import anything under ``api.services.telephony`` at module level
(that package loads every provider, and the providers import ``api.db``).
"""

from typing import Any, Mapping


def credential_value(credentials: Any, field: str) -> Any:
    """One field of a stored credentials document, or ``None`` when it is not set.

    "Not set" is one rule for everything that reads a stored credential: the
    document is not a JSON object (a legacy row stored double-encoded is a string),
    or the field is absent, null or the empty string.
    """
    if not isinstance(credentials, Mapping):
        return None
    value = credentials.get(field)
    return None if value in (None, "") else value
