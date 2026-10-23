"""
Campaign service exceptions.
"""

from api.services.call_concurrency.errors import CONCURRENT_CALL_LIMIT


class ConcurrentSlotAcquisitionError(Exception):
    """Raised when admission refuses a campaign call: no free slot within the timeout (``reason``
    "concurrent_call_limit") or the slot backend is down (``reason`` "admission_backend_unavailable")."""

    def __init__(
        self,
        organization_id: int,
        campaign_id: int,
        wait_time: float,
        reason: str = CONCURRENT_CALL_LIMIT,
    ):
        self.organization_id = organization_id
        self.campaign_id = campaign_id
        self.wait_time = wait_time
        self.reason = reason
        super().__init__(
            f"Failed to acquire concurrent slot for org {organization_id}, "
            f"campaign {campaign_id}: {reason} after waiting {wait_time:.1f}s"
        )


class CampaignRateLimitTimeout(Exception):
    """Temporary dial-rate contention; return the undispatched contact to the queue."""
