"""VOZ-N0-29 (VOZ-AC-B4-46b): POST /inbound/fallback verifies its signature against the configuration its account id names, and answers a
database outage during that lookup with the Spanish unavailable sentence.

The fakes and request builders are shared: api.tests.support.carrier_callbacks and this folder's conftest.py.
"""

import re
from unittest.mock import AsyncMock, patch

import asyncpg.exceptions
import pytest
import sqlalchemy.exc
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors.telephony_errors import CALLER_MESSAGES_ES
from api.routes.telephony import router
from api.tests.support.carrier_callbacks import (
    BASE,
    BODY_MARKER,
    CALLER,
    FALLBACK_PATH,
    NOT_AN_OBJECT,
    OTHER_TOKEN,
    TWILIO_SPANISH_SAY,
    UNAVAILABLE_ES,
    _assert_b028,
    _fallback_form,
    _post_fallback,
    _route_for_twilio,
    _spoken,
    _twilio_signature,
)

# ---------------------------------------------------------------------------------------------------------
# /inbound/fallback
# ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def fallback_provider(world):
    """The provider the matched config builds: real Twilio, so the real signature check runs."""
    _route_for_twilio(world)
    with patch(
        "api.services.telephony.callback_auth.get_telephony_provider_by_id",
        new=AsyncMock(return_value=world.provider),
    ) as built:
        yield built


@pytest.mark.parametrize("signature", ["missing", "forged"])
def test_fallback_bad_signature_is_401_and_moves_nothing(
    client, world, fallback_provider, signature
):
    before = world.snapshot()

    response = _post_fallback(client, _fallback_form(), signature=signature)

    _assert_b028(response)
    assert world.snapshot() == before


def test_fallback_signed_with_another_token_is_401(client, world, fallback_provider):
    response = _post_fallback(client, _fallback_form(), token=OTHER_TOKEN)

    _assert_b028(response)


def test_fallback_unknown_account_is_401(client, world, fallback_provider):
    world.db.route = None

    response = _post_fallback(client, _fallback_form(account_sid="ACunknown"))

    _assert_b028(response)
    fallback_provider.assert_not_awaited()


def test_fallback_that_fails_to_build_its_provider_is_401(client, world):
    _route_for_twilio(world)
    with patch(
        "api.services.telephony.callback_auth.get_telephony_provider_by_id",
        new=AsyncMock(side_effect=ValueError("config parked")),
    ):
        response = _post_fallback(client, _fallback_form())

    _assert_b028(response)


def test_fallback_with_no_body_is_401_not_500(client, world, fallback_provider):
    response = client.post(FALLBACK_PATH, headers={"User-Agent": "TwilioProxy/1.1"})

    _assert_b028(response)


def test_fallback_from_a_provider_nobody_recognises_is_401(
    client, world, fallback_provider
):
    response = client.post(FALLBACK_PATH, data={"Hello": "world"})

    _assert_b028(response)


def test_fallback_signed_answers_in_spanish(client, world, fallback_provider):
    response = _post_fallback(client, _fallback_form())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert re.search(re.escape(TWILIO_SPANISH_SAY) + r"([^<]+)</Say>", response.text)
    spoken = re.search(r"<Say[^>]*>([^<]+)</Say>", response.text).group(1)
    assert spoken in CALLER_MESSAGES_ES.values()
    assert "Sorry" not in response.text and "temporarily" not in response.text


@pytest.mark.parametrize("signature", ["valid", "forged", "missing"])
def test_fallback_does_not_log_headers_or_body(
    client, world, fallback_provider, log_lines, signature
):
    _post_fallback(client, _fallback_form(), signature=signature)

    text = "\n".join(log_lines).lower()
    assert text, "the sink saw nothing: the test would pass for the wrong reason"
    assert CALLER not in text and "612345678" not in text
    assert BODY_MARKER not in text
    # No header is dumped: not a header sent by the request, not a value, not the signature itself. (The
    # provider's own warning may name the header it expected when it is missing; that names none of the request.)
    signature_sent = {
        "valid": _twilio_signature(BASE + FALLBACK_PATH, _fallback_form()),
        "forged": "Zm9yZ2Vk",
        "missing": None,
    }[signature]
    for leaked in (
        "x-custom-probe",
        "custom-header-value",
        "twilioproxy",
        signature_sent,
    ):
        assert leaked is None or leaked.lower() not in text


def test_fallback_for_a_provider_nobody_recognises_does_not_log_headers_or_body(
    client, world, fallback_provider, log_lines
):
    client.post(
        FALLBACK_PATH,
        data={"Marker": BODY_MARKER, "From": CALLER},
        headers={"X-Custom-Probe": "custom-header-value"},
    )

    text = "\n".join(log_lines).lower()
    assert text
    assert CALLER not in text and BODY_MARKER not in text
    assert "x-custom-probe" not in text and "custom-header-value" not in text


def _db_error(cls, *, invalidated: bool):
    return cls(
        "SELECT 1", {}, Exception("driver error"), connection_invalidated=invalidated
    )


# What this stack raises when Postgres is down, restarting, saturated or drops the connection.
LOOKUP_OUTAGES = {
    "connection_refused": lambda: ConnectionRefusedError("connection refused"),
    "builtin_timeout": lambda: TimeoutError(),
    "pool_exhausted": lambda: sqlalchemy.exc.TimeoutError("QueuePool limit reached"),
    "interface_error": lambda: _db_error(
        sqlalchemy.exc.InterfaceError, invalidated=False
    ),
    "connection_invalidated": lambda: _db_error(
        sqlalchemy.exc.OperationalError, invalidated=True
    ),
    "cannot_connect_now": lambda: asyncpg.exceptions.CannotConnectNowError(
        "the database system is starting up"
    ),
    "too_many_connections": lambda: asyncpg.exceptions.TooManyConnectionsError(
        "too many connections"
    ),
    "admin_shutdown": lambda: asyncpg.exceptions.AdminShutdownError("shutting down"),
    "connection_lost": lambda: asyncpg.exceptions.PostgresConnectionError("lost"),
}


def _lookup_raises(world, stage: str, error: Exception):
    """Make the fallback's credential lookup raise ``error`` at ``stage``: the inbound route or the configuration."""
    _route_for_twilio(world)
    if stage == "route":
        world.db.find_inbound_route_by_account = AsyncMock(side_effect=error)
        return patch(
            "api.services.telephony.callback_auth.get_telephony_provider_by_id",
            new=AsyncMock(return_value=world.provider),
        )
    return patch(
        "api.services.telephony.callback_auth.get_telephony_provider_by_id",
        new=AsyncMock(side_effect=error),
    )


@pytest.mark.parametrize("stage", ["route", "configuration"])
@pytest.mark.parametrize("outage", sorted(LOOKUP_OUTAGES))
def test_fallback_during_a_database_outage_speaks_the_unavailable_sentence(
    client, world, log_lines, stage, outage
):
    # Twilio calls the fallback exactly when /inbound/run failed: in an outage the caller hears the Spanish
    # sentence, not the carrier's English application error. Nothing is written and nothing personal is logged.
    before = world.snapshot()
    with _lookup_raises(world, stage, LOOKUP_OUTAGES[outage]()):
        response = _post_fallback(client, _fallback_form())

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/xml")
    assert _spoken(response.text) == [UNAVAILABLE_ES]
    assert world.snapshot() == before
    text = "\n".join(log_lines).lower()
    assert "ca-fallback" in text
    assert CALLER not in text and "612345678" not in text and BODY_MARKER not in text
    assert "custom-header-value" not in text and "twilioproxy" not in text


@pytest.mark.parametrize(
    "error",
    [
        lambda: _db_error(sqlalchemy.exc.DataError, invalidated=False),
        lambda: _db_error(sqlalchemy.exc.ProgrammingError, invalidated=False),
        lambda: _db_error(sqlalchemy.exc.OperationalError, invalidated=False),
        lambda: ValueError("config parked"),
        lambda: AttributeError("'int' object has no attribute 'upper'"),
    ],
    ids=["data_error", "programming_error", "operational_alive", "value", "attribute"],
)
@pytest.mark.parametrize("stage", ["route", "configuration"])
def test_fallback_lookup_error_that_is_not_an_outage_stays_the_401(
    client, world, stage, error
):
    before = world.snapshot()
    with _lookup_raises(world, stage, error()):
        response = _post_fallback(client, _fallback_form())

    _assert_b028(response)
    assert world.snapshot() == before


def test_cloudonix_fallback_during_a_database_outage_answers_xml_not_a_tuple(
    client, world
):
    # RUL-r1-31 (e): CloudonixProvider.generate_error_response returned a (Response, media type) tuple, which the
    # route served as JSON. The outage answer is the provider's spoken XML, as for every other provider.
    from types import SimpleNamespace

    from api.services.telephony.providers.cloudonix.provider import CloudonixProvider

    normalized = SimpleNamespace(
        provider="cloudonix",
        account_id="acme.cloudonix.net",
        to_number="+34911000111",
        to_country="ES",
        call_id="CA-fallback",
    )
    with (
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=CloudonixProvider),
        ),
        patch(
            "api.services.telephony.callback_auth.normalize_webhook_data",
            return_value=normalized,
        ),
        _lookup_raises(world, "route", ConnectionRefusedError("connection refused")),
    ):
        response = _post_fallback(client, _fallback_form())

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/xml")
    assert _spoken(response.text) == [UNAVAILABLE_ES]


@pytest.mark.parametrize("body", NOT_AN_OBJECT)
def test_fallback_with_a_json_body_that_is_not_an_object_is_401_not_500(world, body):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)
    before = world.snapshot()

    response = client.post(
        FALLBACK_PATH, content=body, headers={"content-type": "application/json"}
    )

    _assert_b028(response)
    assert world.snapshot() == before


def test_fallback_with_a_malformed_unsigned_body_is_401(
    client, world, fallback_provider
):
    before = world.snapshot()

    response = client.post(
        FALLBACK_PATH,
        json={**_fallback_form(), "ToCountry": 123, "To": 34911000111},
        headers={"User-Agent": "TwilioProxy/1.1"},
    )

    _assert_b028(response)
    assert world.snapshot() == before
