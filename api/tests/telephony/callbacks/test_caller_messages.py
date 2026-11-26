"""VOZ-N0-29 (VOZ-AC-B4-46b): what a caller or a transfer destination hears is Spanish, from the agreed sentences.

The fakes and request builders are shared: api.tests.support.carrier_callbacks and this folder's conftest.py.
"""

import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from api.errors.telephony_errors import (
    CALL_NOT_ATTENDED_MESSAGE_ES,
    CALLER_MESSAGES_ES,
    TELEPHONY_ERROR_MESSAGES,
    TelephonyError,
)
from api.services.telephony.providers.cloudonix.provider import CloudonixProvider
from api.services.telephony.providers.exotel.provider import ExotelProvider
from api.services.telephony.providers.plivo.provider import PlivoProvider
from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.services.telephony.providers.vobiz.provider import VobizProvider
from api.services.telephony.providers.vonage.provider import VonageProvider
from api.tests.support.carrier_callbacks import (
    API_ROOT,
    CALLED,
    CALLER,
    CAPACITY_ES,
    TRANSFER_ID,
    TWILIO_SPANISH_SAY,
    UNAVAILABLE_ES,
    VONAGE_APP,
    _spoken,
    _twilio,
)

# ---------------------------------------------------------------------------------------------------------
# What the caller hears
# ---------------------------------------------------------------------------------------------------------


def test_caller_messages_are_exactly_the_two_agreed_sentences():
    assert set(CALLER_MESSAGES_ES.values()) == {
        "Lo sentimos, en este momento no podemos atender su llamada. Por favor, inténtelo de nuevo más tarde.",
        "Lo sentimos, este número no está disponible en este momento. Por favor, inténtelo más tarde.",
    }


def test_every_error_has_a_caller_message_and_capacity_ones_say_capacity():
    assert set(CALLER_MESSAGES_ES) == set(TelephonyError) - {TelephonyError.VALID}
    capacity = {
        TelephonyError.CONCURRENT_CALL_LIMIT,
        TelephonyError.QUOTA_EXCEEDED,
        TelephonyError.ADMISSION_BACKEND_UNAVAILABLE,
    }
    for error in capacity:
        assert CALLER_MESSAGES_ES[error] == CAPACITY_ES
    for error in set(CALLER_MESSAGES_ES) - capacity:
        assert CALLER_MESSAGES_ES[error] == UNAVAILABLE_ES


PROVIDER_CLASSES = {
    "twilio": TwilioProvider,
    "cloudonix": CloudonixProvider,
    "exotel": ExotelProvider,
    "plivo": PlivoProvider,
    "vobiz": VobizProvider,
    "vonage": VonageProvider,
}


@pytest.mark.parametrize("provider", sorted(PROVIDER_CLASSES))
@pytest.mark.parametrize(
    "error", [e for e in TelephonyError if e != TelephonyError.VALID]
)
def test_validation_error_is_spoken_in_spanish_from_the_table(provider, error):
    cls = PROVIDER_CLASSES[provider]

    body = cls.generate_validation_error_response(error).body.decode("utf-8")

    assert _spoken(body) == [CALLER_MESSAGES_ES[error]]
    assert TELEPHONY_ERROR_MESSAGES[error] not in body


@pytest.mark.parametrize("provider", sorted(PROVIDER_CLASSES))
def test_validation_error_given_as_its_string_value_is_spoken_from_the_table(
    provider,
):
    cls = PROVIDER_CLASSES[provider]

    body = cls.generate_validation_error_response("QUOTA_EXCEEDED").body.decode("utf-8")

    assert _spoken(body) == [CAPACITY_ES]


def test_vonage_inbound_run_refusal_is_an_ncco_that_speaks_the_table_sentence(
    client, world
):
    # Vonage fetches /inbound/run as its answer URL and reads the answer as NCCO JSON: a TwiML <Hangup/> there
    # is an error on Vonage's side and the caller hears nothing.
    normalized = SimpleNamespace(
        provider="vonage",
        direction="inbound",
        to_number=CALLED,
        from_number=CALLER,
        to_country="ES",
        from_country="ES",
        account_id=VONAGE_APP,
        call_id="vonage-uuid",
    )
    with (
        patch("api.routes.telephony.normalize_webhook_data", return_value=normalized),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=VonageProvider),
        ),
    ):
        response = client.get(
            "/telephony/inbound/run", params={"uuid": "vonage-uuid", "to": "34911"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    talks = [a for a in response.json() if a["action"] == "talk"]
    assert [(a["text"], a["language"]) for a in talks] == [(UNAVAILABLE_ES, "es-ES")]


# The NCCO actions Vonage accepts; the call ends when the actions run out (there is no "hangup" action).
NCCO_ACTIONS = {
    "talk",
    "stream",
    "input",
    "record",
    "conversation",
    "connect",
    "notify",
    "wait",
}


@pytest.mark.parametrize(
    "build",
    [
        lambda: VonageProvider.generate_error_response("SYSTEM_UNAVAILABLE", "x"),
        lambda: VonageProvider.generate_validation_error_response(
            TelephonyError.QUOTA_EXCEEDED
        ),
    ],
    ids=["error", "validation"],
)
def test_vonage_error_ncco_has_only_actions_vonage_accepts(build):
    actions = [a["action"] for a in json.loads(build().body)]

    assert actions == ["talk"]
    assert set(actions) <= NCCO_ACTIONS


@pytest.mark.parametrize("provider", sorted(PROVIDER_CLASSES))
@pytest.mark.parametrize(
    "error_type",
    ["SYSTEM_UNAVAILABLE", "auth_failed", "not_configured", "invalid_number", "other"],
)
def test_generic_error_is_spoken_in_spanish_and_never_speaks_the_debug_text(
    provider, error_type
):
    cls = PROVIDER_CLASSES[provider]
    debug_text = "Debug detail for the operator"

    response = cls.generate_error_response(error_type, debug_text)
    response = response[0] if isinstance(response, tuple) else response
    body = response.body.decode("utf-8")

    spoken = _spoken(body)
    assert spoken and all(t in CALLER_MESSAGES_ES.values() for t in spoken)
    assert "Debug detail" not in body


@pytest.mark.parametrize(
    ("provider", "marker"),
    [
        ("twilio", TWILIO_SPANISH_SAY),
        ("cloudonix", '<Say language="es-ES">'),
        ("plivo", 'language="es-ES"'),
        ("vobiz", 'language="es-ES"'),
        ("vonage", '"language": "es-ES"'),
        pytest.param(
            "exotel",
            None,
            marks=pytest.mark.skip(
                reason="GATED exotel_language_unverified: Exotel's <Say> language and voice attributes are not "
                "documented offline and there is no account to try them (R9), so none is guessed; VOZ-N0-40 Step 2"
            ),
        ),
    ],
)
def test_spoken_errors_name_the_language(provider, marker):
    cls = PROVIDER_CLASSES[provider]

    response = cls.generate_error_response("SYSTEM_UNAVAILABLE", "x")
    response = response[0] if isinstance(response, tuple) else response

    assert marker in response.body.decode("utf-8")


def test_every_spanish_twilio_say_names_a_voice_that_speaks_es_es():
    # Twilio's default voice ("man") does not list es-ES and "alice" is retired: each Spanish <Say> names the
    # Polly voice, or Twilio reads the sentence with the wrong voice or drops it.
    from api.services.telephony.providers.twilio.routes import _connect_action_twiml

    bodies = {
        "error": TwilioProvider.generate_error_response("SYSTEM_UNAVAILABLE", "x"),
        "validation": TwilioProvider.generate_validation_error_response(
            TelephonyError.CONCURRENT_CALL_LIMIT
        ),
    }
    bodies = {name: r.body.decode("utf-8") for name, r in bodies.items()}
    bodies["connect-action"] = _connect_action_twiml(call_ended=False)

    for name, body in bodies.items():
        says = re.findall(r"<Say\b[^>]*>", body)
        assert says == [TWILIO_SPANISH_SAY], name


TRANSFER_GREETING = "Tiene una llamada transferida. Le conectamos ahora."


# What a caller or a transfer destination may hear, and the names that produce it in the sources.
AUDIBLE_TEXTS = set(CALLER_MESSAGES_ES.values()) | {
    TRANSFER_GREETING,
    CALL_NOT_ATTENDED_MESSAGE_ES,
}


AUDIBLE_PRODUCERS = (
    "caller_message_es(",
    "log_and_get_caller_text(",
    "UNAVAILABLE_MESSAGE_ES",
    "TRANSFER_GREETING_ES",
    "CALL_NOT_ATTENDED_MESSAGE_ES",
)


SOURCE_SPEECH = re.compile(r"<(Say|Speak)\b[^>]*>(.*?)</\1>", re.S)


SOURCE_TALK_TEXT = re.compile(r'"text":\s*([^\n]+)')


# The seam between two adjacent string literals of one implicitly concatenated string.
LITERAL_SEAM = re.compile(r"""["']\s*f?["']""")


def _audible(source: str) -> bool:
    """Whether ``source`` (what a <Say>/<Speak> or a talk "text" holds in the code) can only be an agreed text:
    a literal of AUDIBLE_TEXTS, or one expression of an AUDIBLE_PRODUCERS name."""
    text = LITERAL_SEAM.sub("", source).strip().rstrip(",").strip()
    if text[:1] == "{" and text[-1:] == "}" and text.count("{") == 1:
        text = text[1:-1]
    elif text[:1] in "\"'" and text[-1:] == text[:1]:
        return text[1:-1] in AUDIBLE_TEXTS
    elif text in AUDIBLE_TEXTS:
        return True
    return text.startswith(AUDIBLE_PRODUCERS) or any(
        f"({name}" in text for name in AUDIBLE_PRODUCERS
    )


def test_no_audible_text_outside_the_agreed_set_in_the_telephony_sources():
    """Grep guard: every literal inside <Say>/<Speak> and every talk text in the telephony routes and services is
    one of the agreed Spanish texts (VOZ-AC-B4-46b: no English, no free text)."""
    files = [API_ROOT / "routes" / "telephony.py"] + sorted(
        (API_ROOT / "services" / "telephony").rglob("*.py")
    )
    offenders = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        spoken = [m.group(2) for m in SOURCE_SPEECH.finditer(source)]
        spoken += [m.group(1) for m in SOURCE_TALK_TEXT.finditer(source)]
        offenders += [
            f"{path.relative_to(API_ROOT)}: {text.strip()[:80]}"
            for text in spoken
            if not _audible(text)
        ]
    assert offenders == []


@pytest.mark.parametrize(
    "source",
    [
        "Connecting you now.",
        '"Sorry, an error occurred",',
        "Hola {name}",
        "{escape(message)}",
        "{message}",
        "{UNAVAILABLE_MESSAGE_ES} {name}",
    ],
)
def test_the_audible_text_guard_refuses_free_or_english_text(source):
    assert not _audible(source)


def _twilio_transfer_twiml() -> str:
    """The inline TwiML Twilio's transfer_call sends for the destination leg, captured before it is sent."""
    sent = []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, data=None, **kwargs):
            sent.append(data)
            raise RuntimeError("test: stop after building the request")

    module = "api.services.telephony.providers.twilio.provider"
    with (
        patch(f"{module}.aiohttp.ClientSession", lambda **kwargs: Session()),
        patch(
            f"{module}.get_backend_endpoints",
            new=AsyncMock(return_value=("https://backend.test", "")),
        ),
    ):
        try:
            asyncio.run(
                _twilio().transfer_call(
                    destination="+34911222333",
                    transfer_id=TRANSFER_ID,
                    conference_name="conf-1",
                )
            )
        except Exception:
            pass
    return sent[0]["Twiml"]


def _plivo_transfer_xml() -> str:
    from api.services.telephony.providers.plivo.routes import _conference_xml_response

    return _conference_xml_response("conf-1").body.decode("utf-8")


@pytest.mark.parametrize(
    ("build", "tag"),
    [
        (_twilio_transfer_twiml, TWILIO_SPANISH_SAY),
        (_plivo_transfer_xml, '<Speak language="es-ES">'),
        (
            lambda: CloudonixProvider._conference_join_cxml(
                "conf-1", "https://backend.test/cb"
            ),
            '<Say language="es-ES">',
        ),
    ],
    ids=["twilio", "plivo", "cloudonix"],
)
def test_transfer_greeting_is_the_spanish_sentence_in_a_spanish_voice(build, tag):
    body = build()

    assert re.findall(r"<(?:Say|Speak)\b[^>]*>", body) == [tag]
    assert _spoken(body) == [TRANSFER_GREETING]


@pytest.mark.parametrize("provider", sorted(PROVIDER_CLASSES))
def test_generic_error_logs_the_operator_text(provider, log_lines):
    # The second argument of generate_error_response is for the operator: logged, never spoken.
    cls = PROVIDER_CLASSES[provider]

    cls.generate_error_response("auth_failed", "Debug detail for the operator")

    logged = [line for line in log_lines if "Debug detail for the operator" in line]
    assert len(logged) == 1
    assert "auth_failed" in logged[0] and provider in logged[0]
