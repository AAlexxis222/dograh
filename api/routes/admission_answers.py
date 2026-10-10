"""Each channel's answer to a refused admission (VOZ-AC-B3-65-bis): the one place that maps a refusal's true reason
(a full org, or the slot backend down) to the HTTP, WebSocket, browser signaling and carrier answers. Every
function takes the refusal; a route calls one of them. The audible wording of the carrier answer is VOZ-N0-29's.
"""

from fastapi.responses import JSONResponse

from api.errors.telephony_errors import TelephonyError
from api.services.call_concurrency import (
    AdmissionBackendUnavailableError,
    CallConcurrencyLimitError,
    WorkflowRunSlotAlreadyBoundError,
)
from api.services.runtime.durations import cell_durations


def _backend_down(error: CallConcurrencyLimitError) -> bool:
    return isinstance(error, AdmissionBackendUnavailableError)


def http_response(error: CallConcurrencyLimitError) -> JSONResponse:
    """429 for a full org; 503 with Retry-After (RFC 9110 §10.2.3) when Redis is down. ``detail`` stays a
    string, because clients render it as text; the B0-28 record rides as top-level fields of the same body."""
    if _backend_down(error):
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"{error.reason}: {error.what} (where: {error.where}; hint: {error.hint})",
                **error.failure(),
            },
            headers={"Retry-After": str(cell_durations().admission_retry_after_s)},
        )
    return JSONResponse(
        status_code=429, content={"detail": "Concurrent call limit reached"}
    )


def ws_close(error: CallConcurrencyLimitError) -> dict:
    """WebSocket close kwargs: 1008 for a full org, 1013 (try again later) when Redis is down. A close reason
    holds at most 123 bytes (RFC 6455 §5.5), so it carries code, where and hint, not the whole record."""
    if _backend_down(error):
        return {
            "code": 1013,
            "reason": f"{error.reason} at {error.where}: {error.hint}",
        }
    return {"code": 1008, "reason": "Concurrent call limit reached"}


def client_error(error: CallConcurrencyLimitError) -> dict:
    """Error payload sent to a browser client over the signaling socket."""
    if _backend_down(error):
        return {
            "error_type": error.reason,
            "message": "Service temporarily unavailable",
            **error.failure(),
        }
    return {
        "error_type": "concurrency_limit_exceeded",
        "message": "Concurrent call limit reached",
    }


def bind_failure_response(
    error: CallConcurrencyLimitError | WorkflowRunSlotAlreadyBoundError,
) -> JSONResponse:
    """HTTP answer when binding a run to its slot fails: 409 when the run already holds one (a real duplicate), the
    refusal's own answer (503) when the slot backend did not answer."""
    if isinstance(error, CallConcurrencyLimitError):
        return http_response(error)
    return JSONResponse(
        status_code=409, content={"detail": "Workflow run already has an active call"}
    )


def carrier_error(
    error: CallConcurrencyLimitError | WorkflowRunSlotAlreadyBoundError,
) -> TelephonyError:
    """The carrier answer for an inbound call that got no slot: a slot backend that is down (failing closed), else
    a full org or a run that already holds a slot."""
    if _backend_down(error):
        return TelephonyError.ADMISSION_BACKEND_UNAVAILABLE
    return TelephonyError.CONCURRENT_CALL_LIMIT
