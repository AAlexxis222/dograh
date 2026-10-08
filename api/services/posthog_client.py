from typing import Any, Optional

from loguru import logger
from posthog import Posthog

from api import constants

_posthog_client: Posthog | None = None
POSTHOG_SERVER_GROUP_IDENTIFY_DISTINCT_ID = "server-group-identify"
POSTHOG_ORGANIZATION_GROUP_TYPE = "organization"


def get_posthog() -> Posthog | None:
    """Return the lazily-initialised PostHog client.

    None unless telemetry is explicitly enabled AND an API key is configured:
    no opt-in means no client, so nothing can leave the process. Sentry keeps
    its own upstream rule in app.py (DSN-gated; cells ship without a DSN).
    """
    global _posthog_client
    if (
        _posthog_client is None
        and constants.ENABLE_TELEMETRY
        and constants.POSTHOG_API_KEY
    ):
        _posthog_client = Posthog(
            constants.POSTHOG_API_KEY, host=constants.POSTHOG_HOST
        )
    return _posthog_client


def shutdown_posthog() -> None:
    """Flush queued PostHog messages before a short-lived process exits."""
    client = get_posthog()
    if not client:
        return
    try:
        client.shutdown()
    except Exception:
        logger.exception("Failed to shut down PostHog client")


def flush_posthog() -> None:
    """Flush queued PostHog messages without shutting down the client."""
    client = get_posthog()
    if not client:
        return
    try:
        client.flush()
    except Exception:
        logger.exception("Failed to flush PostHog client")


def capture_event(
    distinct_id: str,
    event: str,
    properties: dict[str, Any] | None = None,
    groups: Optional[dict[str, str]] = None,
) -> None:
    """Fire a PostHog event. Silently no-ops if PostHog is not configured."""
    client = get_posthog()
    if not client:
        return
    try:
        kwargs: dict[str, Any] = {
            "distinct_id": distinct_id,
            "event": event,
            "properties": properties or {},
        }
        if groups:
            kwargs["groups"] = groups
        client.capture(**kwargs)
    except Exception:
        logger.exception(f"Failed to send PostHog event '{event}'")


def group_identify(
    group_type: str,
    group_key: str,
    properties: dict[str, Any],
    *,
    distinct_id: Optional[str] = None,
) -> None:
    """Set PostHog group properties. Silently no-ops if PostHog is not configured."""
    client = get_posthog()
    if not client:
        return
    try:
        client.group_identify(
            group_type,
            group_key,
            properties,
            distinct_id=distinct_id or POSTHOG_SERVER_GROUP_IDENTIFY_DISTINCT_ID,
        )
    except Exception:
        logger.exception("Failed to identify PostHog group")


def set_person_properties(distinct_id: str, properties: dict[str, Any]) -> None:
    """Set PostHog person properties. Silently no-ops if PostHog is not configured."""
    client = get_posthog()
    if not client:
        return
    try:
        client.set(distinct_id=distinct_id, properties=properties)
    except Exception:
        logger.exception("Failed to set PostHog person properties")
