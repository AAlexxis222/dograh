"""Single redaction filter shared by logs, traces and Sentry."""

import re
import traceback

_PATTERNS = [
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), "<redacted:key>"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"), "Bearer <redacted>"),
    (re.compile(r"(wss?|https?)://[^/\s:@]+:[^/\s@]+@"), r"\1://<redacted>@"),
    # E.164: "+" then 8-15 digits, a single space, dot or dash allowed between digits.
    (re.compile(r"(?<![\w+])\+\d(?:[ .-]?\d){7,14}(?!\d)"), "<redacted:phone>"),
    # Spanish national number: 9 digits starting 6-9, optional single spaces.
    # The boundaries keep ISO dates/times, UUIDs and longer numeric ids intact.
    (re.compile(r"(?<![\w-])[6-9]\d{2}(?: ?\d){6}(?!\w)"), "<redacted:phone>"),
]


def known_secret_values() -> set[str]:
    from api.services.configuration.secrets_registry import active_secret_values

    return active_secret_values()


def redact(text: str) -> str:
    for value in known_secret_values():
        text = text.replace(value, "<redacted:secret>")
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact_log_record(record: dict) -> None:
    """loguru patcher: redact the message and fold the exception into it.

    loguru renders traceback source lines at format time, after any patcher
    has run, so redacting only ``record["message"]`` leaves a secret that sits
    in a ``raise`` line. Rendering the traceback here, redacting it and
    clearing ``record["exception"]`` keeps the formatted prefix (timestamp,
    level, run_id, file:line) out of the redaction's reach in every sink.
    """
    message = redact(record["message"])
    exception = record["exception"]
    if exception is not None:
        rendered = "".join(traceback.format_exception(*exception))
        message = f"{message}\n{redact(rendered).rstrip()}"
        record["exception"] = None
    record["message"] = message
