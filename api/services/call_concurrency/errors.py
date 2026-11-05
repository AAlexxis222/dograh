"""Admission refusals and their stable codes (VOZ-AC-B0-28). Data only: each channel's answer to a refusal lives at
the edge, in api/routes/admission_answers.py."""

# The stable ``code`` of each refusal, the same string in every channel's answer and in the campaign error.
CONCURRENT_CALL_LIMIT = "concurrent_call_limit"
ADMISSION_BACKEND_UNAVAILABLE = "admission_backend_unavailable"


class CallConcurrencyLimitError(Exception):
    """Raised when admission refuses a call because no slot is free (``reason`` "concurrent_call_limit")."""

    reason = CONCURRENT_CALL_LIMIT
    # The rest of the VOZ-AC-B0-28 record: what happened, where, what to do. Set by refusals that carry one.
    what: str | None = None
    where: str | None = None
    hint: str | None = None

    def __init__(
        self,
        *,
        organization_id: int,
        source: str,
        wait_time: float,
        max_concurrent: int,
    ):
        self.organization_id = organization_id
        self.source = source
        self.wait_time = wait_time
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Call admission refused for org {organization_id}: {self.reason} "
            f"(source={source}, limit={max_concurrent}, waited={wait_time:.1f}s)"
            + (f" (hint: {self.hint})" if self.hint else "")
        )

    def failure(self) -> dict:
        """The VOZ-AC-B0-28 record of the refusal: stable code, what happened, where, and what to do."""
        return {
            "code": self.reason,
            "reason": self.what,
            "where": self.where,
            "hint": self.hint,
        }


class AdmissionBackendUnavailableError(CallConcurrencyLimitError):
    """VOZ-AC-B3-65-bis: admission fails closed when the slot backend (Redis) cannot answer. The org is not full,
    so every channel answers it differently (HTTP 503, WS 1013, its own carrier code); a caller that catches
    ``CallConcurrencyLimitError`` for "org full" logic must check for this subclass first."""

    reason = ADMISSION_BACKEND_UNAVAILABLE
    what = "the slot backend (Redis) did not answer, so admission failed closed"
    where = "call_concurrency.acquire_slot"
    hint = "check Redis (REDIS_URL), then retry the call"
