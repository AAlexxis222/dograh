"""
Telephony error constants and messages for inbound call validation.
Centralizes error handling across all telephony providers.
"""

from enum import Enum

from loguru import logger

from api.errors.failure import DograhFailure, ErrorSource, ErrorType, log_failure


class TelephonyError(Enum):
    """Telephony validation error types"""

    PROVIDER_MISMATCH = "PROVIDER_MISMATCH"
    WORKFLOW_NOT_FOUND = "WORKFLOW_NOT_FOUND"
    ACCOUNT_VALIDATION_FAILED = "ACCOUNT_VALIDATION_FAILED"
    PHONE_NUMBER_NOT_CONFIGURED = "PHONE_NUMBER_NOT_CONFIGURED"
    SIGNATURE_VALIDATION_FAILED = "SIGNATURE_VALIDATION_FAILED"
    CONCURRENT_CALL_LIMIT = "CONCURRENT_CALL_LIMIT"
    # Admission failed closed: the slot backend (Redis) did not answer (VOZ-AC-B3-65-bis).
    ADMISSION_BACKEND_UNAVAILABLE = "ADMISSION_BACKEND_UNAVAILABLE"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    GENERAL_AUTH_FAILED = "GENERAL_AUTH_FAILED"
    # The carrier is disabled until its signing is verified (carrier_policy.UNVERIFIED_CARRIERS, VOZ-N0-40).
    CARRIER_SIGNING_UNVERIFIED = "CARRIER_SIGNING_UNVERIFIED"
    VALID = "VALID"


# Error messages for organizations (debugging-focused). Written to logs and failure records only: what a caller
# HEARS is CALLER_MESSAGES_ES below.
TELEPHONY_ERROR_MESSAGES = {
    TelephonyError.PROVIDER_MISMATCH: "Configuration error: This phone number is configured for a different telephony provider. Please check your dashboard settings and update your webhook URL configuration.",
    TelephonyError.WORKFLOW_NOT_FOUND: "Workflow not found. Please verify the workflow ID in your webhook URL is correct and the workflow exists in your dashboard.",
    TelephonyError.ACCOUNT_VALIDATION_FAILED: "Authentication error: Account credentials do not match. Please verify your account SID configuration in the dashboard matches your telephony provider settings.",
    TelephonyError.PHONE_NUMBER_NOT_CONFIGURED: "Phone number not configured: This number is not set up for inbound calls in your account. Please add this number to your telephony configuration.",
    TelephonyError.SIGNATURE_VALIDATION_FAILED: "Security error: Webhook signature validation failed. Please verify your auth token configuration and ensure requests are coming from your telephony provider.",
    TelephonyError.CONCURRENT_CALL_LIMIT: "Service temporarily unavailable: Your account has reached its concurrent call limit. Please try again later.",
    TelephonyError.ADMISSION_BACKEND_UNAVAILABLE: "Service temporarily unavailable. Please try again later.",
    TelephonyError.QUOTA_EXCEEDED: "Service temporarily unavailable: Your account has exceeded usage limits. Please contact your administrator or upgrade your plan to continue receiving calls.",
    TelephonyError.GENERAL_AUTH_FAILED: "Authentication failed: Please check your webhook URL configuration and ensure your telephony provider settings match your dashboard configuration.",
    TelephonyError.CARRIER_SIGNING_UNVERIFIED: "Carrier disabled: this platform cannot verify this carrier's webhook signatures yet, so it refuses its calls. Please use another telephony provider.",
}


# What the caller hears (VOZ-AC-B4-46b): two Spanish sentences, never the debugging text above. Capacity
# refusals say "we cannot take your call now"; everything else says the number is not available.
CALLER_LANGUAGE = "es-ES"
# The voice of every Spanish Twilio <Say>: the default voice ("man") does not list es-ES and "alice" is retired.
TWILIO_CALLER_VOICE = "Polly.Lucia"
_CAPACITY_MESSAGE_ES = "Lo sentimos, en este momento no podemos atender su llamada. Por favor, inténtelo de nuevo más tarde."
UNAVAILABLE_MESSAGE_ES = "Lo sentimos, este número no está disponible en este momento. Por favor, inténtelo más tarde."
# Twilio connect action: played when the media socket ended and the call did not finish normally.
CALL_NOT_ATTENDED_MESSAGE_ES = (
    "Ahora mismo no podemos atender tu llamada. "
    "Por favor, inténtalo de nuevo en unos minutos."
)
# What the transfer destination hears when it answers, before it joins the conference. Not a caller error, so
# not in CALLER_MESSAGES_ES.
TRANSFER_GREETING_ES = "Tiene una llamada transferida. Le conectamos ahora."
_QUOTA_ERRORS = {
    TelephonyError.CONCURRENT_CALL_LIMIT,
    TelephonyError.QUOTA_EXCEEDED,
}
_CAPACITY_ERRORS = _QUOTA_ERRORS | {TelephonyError.ADMISSION_BACKEND_UNAVAILABLE}
CALLER_MESSAGES_ES = {
    error: _CAPACITY_MESSAGE_ES if error in _CAPACITY_ERRORS else UNAVAILABLE_MESSAGE_ES
    for error in TelephonyError
    if error != TelephonyError.VALID
}


def caller_message_es(error) -> str:
    """The sentence a caller hears for ``error`` (a TelephonyError or its value). Anything else, such as the
    free-form types passed to ``generate_error_response``, gets the "not available" sentence."""
    try:
        return CALLER_MESSAGES_ES[TelephonyError(error)]
    except (KeyError, ValueError):
        return UNAVAILABLE_MESSAGE_ES


def log_and_get_caller_text(error_type, message: str, *, provider: str) -> str:
    """Log ``message`` as a warning, then return what the caller hears from ``provider``'s
    ``generate_error_response``: ``caller_message_es(error_type)``.

    ``message`` is the operator-facing text of the same error: logged with ``error_type`` and ``provider``, never
    spoken. Callers pass fixed operator text, never caller data.
    """
    logger.warning(
        f"telephony error response: provider={provider} error_type={error_type} message={message}"
    )
    return caller_message_es(error_type)


def failure_from_telephony_error(
    error: TelephonyError,
    *,
    provider: str | None = None,
) -> DograhFailure | None:
    """Map an inbound validation result onto the stable failure taxonomy."""

    if not isinstance(error, TelephonyError):
        try:
            error = TelephonyError(error)
        except (TypeError, ValueError):
            provider_code = (provider or "telephony").replace("_", "-")
            return DograhFailure(
                source=ErrorSource.TELEPHONY,
                type=ErrorType.SYSTEM_ERROR,
                code=f"{provider_code}-unknown-validation-error",
                internal_message=f"Unknown inbound telephony validation result: {error}",
                external_message="Dograh could not process the inbound telephony validation result.",
                provider=provider,
                retryable=None,
                context={
                    "notification_eligible": False,
                    "tenant_attribution_trusted": False,
                },
            )
    if error == TelephonyError.VALID:
        return None

    error_type = (
        ErrorType.QUOTA_ERROR if error in _QUOTA_ERRORS else ErrorType.CONFIG_ERROR
    )
    # Ours, not the tenant's, and it clears when Redis answers again.
    backend_down = error == TelephonyError.ADMISSION_BACKEND_UNAVAILABLE
    if backend_down:
        error_type = ErrorType.SYSTEM_ERROR
    provider_code = (provider or "telephony").replace("_", "-")
    external_message = TELEPHONY_ERROR_MESSAGES.get(
        error, TELEPHONY_ERROR_MESSAGES[TelephonyError.GENERAL_AUTH_FAILED]
    )
    return DograhFailure(
        source=ErrorSource.TELEPHONY,
        type=error_type,
        code=f"{provider_code}-{error.value.lower().replace('_', '-')}",
        internal_message=f"Inbound telephony validation failed: {error.value}",
        external_message=external_message,
        provider=provider,
        error_owner="operator" if backend_down else "user",
        retryable=backend_down,
        context={
            # A failed signature can be sent by an attacker and must not become a
            # tenant notification until attribution has been established.
            "notification_eligible": error
            != TelephonyError.SIGNATURE_VALIDATION_FAILED,
            "tenant_attribution_trusted": error
            != TelephonyError.SIGNATURE_VALIDATION_FAILED,
        },
    )


def log_telephony_error(
    error: TelephonyError,
    *,
    provider: str | None = None,
    **context,
) -> None:
    failure = failure_from_telephony_error(error, provider=provider)
    if failure is not None:
        log_failure(failure, **context)
