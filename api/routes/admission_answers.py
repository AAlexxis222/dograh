"""Each channel's answer to a refused admission (VOZ-AC-B3-65-bis): the one place that maps a refusal's true reason
(a full org, or the slot backend down) to the HTTP, WebSocket, browser signaling, carrier and ARI answers. Every
function takes the refusal; a route calls one of them. The audible wording of the carrier answer is VOZ-N0-29's.
"""

from fastapi.responses import JSONResponse

from api.errors.telephony_errors import TelephonyError
from api.services.call_concurrency import (
    AdmissionBackendUnavailableError,
    CallConcurrencyLimitError,
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


def carrier_error(error: CallConcurrencyLimitError) -> TelephonyError:
    """The carrier answer for a refused inbound call: a full org, or (failing closed) a slot backend that is down."""
    if _backend_down(error):
        return TelephonyError.ADMISSION_BACKEND_UNAVAILABLE
    return TelephonyError.CONCURRENT_CALL_LIMIT


def ari_hangup_note(
    error: CallConcurrencyLimitError, organization_id: int, channel_id: str
) -> str:
    """ARI answers every refusal by hanging the inbound channel up; this is the line logged with it, which names
    the true reason."""
    return (
        f"[ARI org={organization_id}] Call admission refused "
        f"({error.reason}); hanging up inbound channel {channel_id}"
    )
