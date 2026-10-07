"""Redaction for loguru log records: secrets and phone numbers never reach a sink.

Scope: log records only. Sentry's ``before_send`` and traces are a follow-up.
Sentry's auto-enabled LoguruIntegration receives a message-only event (no
structured frames) because the redacted traceback is folded into ``message``;
that is the privacy-first choice, since the raw exception values would
otherwise reach Sentry unredacted.
"""

import os
import re
import traceback
from collections.abc import Iterable

from api.errors.failure import fold_for_gate, redact_credentials

_PHONE = "<redacted:phone>"

# (needles, pattern, replacement). ``needles`` are lowercase substrings, at least
# one of which must be in the casefolded text for the pattern to run: a regex
# scan costs ~10us a line and almost no log line carries a phone or a URL
# credential, so the substring test (memchr-fast) is what keeps this off the
# call loop's budget. Each needle set is a necessary condition for its pattern.
_PATTERNS = [
    # Userinfo of a URL with ANY scheme (postgresql+asyncpg://u:p@, redis://:p@).
    # redact_credentials already covers http(s)/ws(s); this covers the rest.
    (
        ("@",),
        re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s@]*@"),
        r"\1<redacted>@",
    ),
    (("sk-", "sk_"), re.compile(r"\bsk[-_][A-Za-z0-9_\-]{8,}"), "<redacted:key>"),
    # sip:/tel: URIs: every digit of the user part is a phone (or extension).
    (
        ("sip",),
        re.compile(r"(?i)\b(sips?:)(?:%2B|\+)?\d(?:[.\-()]*\d)*(?=@)"),
        rf"\1{_PHONE}",
    ),
    (
        ("tel:",),
        re.compile(r"(?i)\b(tel:)(?:%2B|\+)?\d(?:[.\-()]*\d)*"),
        rf"\1{_PHONE}",
    ),
    # International: "+", "00", or a url-encoded "+" ("%2B", "%252B") then 8-15
    # digits (E.164), each separated by at most two of space . - ( ), so
    # "(+34) 612 345 678" and "+1 (415) 555-2671" go whole, prefix included.
    # The lookbehind keeps the "+02:00" of a timestamp out (it follows a digit);
    # the encoded form needs none, as "%22%2B34..." and "Hello%20%2B34..." are
    # common in logged URLs and bodies.
    (
        ("+", "%2b", "%252b", "00"),
        re.compile(
            r"(?:(?<![\w+])\(?(?:\+|00(?=[1-9]))|\(?%(?:25)?2[Bb])"
            r"(?:[ .\-()]{0,2}\d){8,15}(?!\d)"
        ),
        _PHONE,
    ),
    # Spanish number, with or without the bare "34" country code: 9 digits
    # starting 6-9, single space/hyphen separators allowed; dots only in the
    # grouped shapes 3-2-2-2 and 3-3-3, so decimals ("712.345678") stay. The
    # lookbehind keeps out ISO dates, UUID segments, longer ids and decimal
    # fractions ("0.712345678"). "+" is not in it: "call+me+at+612345678" is a
    # form-encoded text, and a real "+34..." was already taken above.
    # Accepted false positive: a standalone 9-digit id starting 6-9
    # (workflow_run_id=712345678) is redacted; privacy first.
    (
        ("6", "7", "8", "9"),
        re.compile(
            r"(?<![\w.\-%])(?:34)?[6-9]\d{2}"
            r"(?:(?:[ \-]?\d){6}|(?:\.\d{2}){3}|(?:\.\d{3}){2})(?!\w)"
        ),
        _PHONE,
    ),
]


_SECRET_ENV_MARKERS: tuple[str, ...] = ("SECRET", "PASSWORD", "TOKEN")
# Shorter values ("true", "dev") are flags, not secrets; redacting them would
# mangle every log line that contains the word.
_MIN_SECRET_LENGTH = 8


def _is_secret_env_name(name: str) -> bool:
    """Secret-named variable: a marker anywhere in the name (``AWS_SECRET_ACCESS_KEY``,
    ``DB_PASSWORD_FILE``), or a ``_KEY`` suffix (``OPENAI_API_KEY``)."""
    upper = name.upper()
    return upper.endswith("_KEY") or any(
        marker in upper for marker in _SECRET_ENV_MARKERS
    )


def active_secret_values() -> set[str]:
    """Secret values this process holds in memory right now, for log redaction.

    Today: the values of secret-named environment variables. The log patcher
    binds this set once, at ``setup_logging``, so a secret loaded later (the
    decrypted database credentials of VOZ-N0-23) is not redacted until the
    credential box rebuilds that binding with a refreshed set.
    """
    return {
        value
        for name, value in os.environ.items()
        if _is_secret_env_name(name) and len(value) >= _MIN_SECRET_LENGTH
    }


def known_secret_values() -> tuple[str, ...]:
    """Secret values held by this process, longest first, so a secret that is a
    prefix of another cannot leave the tail of the longer one in clear."""
    return tuple(sorted(active_secret_values(), key=len, reverse=True))


def redact(text: str, secrets: Iterable[str] | None = None) -> str:
    """Redact ``text``. ``secrets`` must be ordered longest first (the output of
    ``known_secret_values``); ``None`` reads them now. Hot paths compute them
    once and pass them in."""
    if secrets is None:
        secrets = known_secret_values()
    for value in secrets:
        text = text.replace(value, "<redacted:secret>")
    text = redact_credentials(text)
    folded = fold_for_gate(text)
    for needles, pattern, repl in _PATTERNS:
        if any(needle in folded for needle in needles):
            text = pattern.sub(repl, text)
    return text


def redact_log_record(record: dict, secrets: Iterable[str] | None = None) -> None:
    """loguru patcher: redact the message and fold the exception into it.

    loguru renders traceback source lines at format time, after any patcher
    has run, so redacting only ``record["message"]`` leaves a secret that sits
    in a ``raise`` line. Rendering the traceback here, redacting it and
    clearing ``record["exception"]`` keeps the formatted prefix (timestamp,
    level, run_id, file:line) out of the redaction's reach in every sink.
    """
    message = redact(record["message"], secrets)
    exception = record["exception"]
    if exception is not None:
        rendered = "".join(traceback.format_exception(*exception))
        message = f"{message}\n{redact(rendered, secrets).rstrip()}"
        record["exception"] = None
    record["message"] = message
