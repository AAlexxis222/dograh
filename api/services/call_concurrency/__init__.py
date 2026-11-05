"""Org-wide call concurrency control for every call type (telephony, WebRTC,
agent streams, campaigns).

``service.py`` is the facade routes and tasks go through; ``slots.py`` owns the
slot-lease lifecycle in Redis, ``errors.py`` the admission refusals, and
``rate_limiter.py`` the per-second rate limiting (plus the upstream slot method
names, which forward to the slot store).
"""

from api.services.call_concurrency.errors import (
    AdmissionBackendUnavailableError,
    CallConcurrencyLimitError,
)
from api.services.call_concurrency.service import (
    CallConcurrencyService,
    CallConcurrencySlot,
    WorkflowRunSlotAlreadyBoundError,
    call_concurrency,
)
from api.services.call_concurrency.slots import SlotBackendError

__all__ = [
    "AdmissionBackendUnavailableError",
    "CallConcurrencyLimitError",
    "CallConcurrencyService",
    "CallConcurrencySlot",
    "SlotBackendError",
    "WorkflowRunSlotAlreadyBoundError",
    "call_concurrency",
]
