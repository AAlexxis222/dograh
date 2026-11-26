"""VOZ-N0-29 (VOZ-AC-B4-46b): the transfer-result callbacks (Twilio and Telnyx) verify their signature against the credentials of the transfer's run and fail closed.

The fakes and request builders are shared: api.tests.support.carrier_callbacks and this folder's conftest.py.
"""

import base64
import time

import nacl.signing
import pytest

from api.services.telephony.transfer_event_protocol import TransferEventType
from api.tests.support.carrier_callbacks import (
    FAKE,
    OTHER_TOKEN,
    TELNYX_KEY,
    TRANSFER_ID,
    _assert_b028,
    _cloudonix,
    _context,
    _post_telnyx,
    _post_transfer_result,
    _telnyx,
    _telnyx_body,
)

# ---------------------------------------------------------------------------------------------------------
# /transfer-result/{transfer_id} (Twilio StatusCallback of the transfer leg)
# ---------------------------------------------------------------------------------------------------------


def test_transfer_result_unsigned_forgery_is_401_and_moves_nothing(client, world):
    before = world.snapshot()

    response = _post_transfer_result(client, signed=False)

    _assert_b028(response)
    assert world.snapshot() == before
    assert world.manager.published == []


def test_transfer_result_signature_of_another_token_is_401(client, world):
    before = world.snapshot()

    response = _post_transfer_result(client, signed=True, token=OTHER_TOKEN)

    _assert_b028(response)
    assert world.snapshot() == before


@pytest.mark.parametrize(
    "scenario",
    [
        "no_transfer_context",
        "context_without_workflow_run_id",
        "run_not_found",
        "workflow_not_found",
        "provider_cannot_be_built",
        "run_belongs_to_another_provider",
    ],
)
def test_transfer_result_without_a_credential_context_is_401_even_if_signed(
    client, world, scenario
):
    if scenario == "no_transfer_context":
        world.manager.contexts.clear()
    elif scenario == "context_without_workflow_run_id":
        world.manager.contexts[TRANSFER_ID] = _context(workflow_run_id=None)
    elif scenario == "run_not_found":
        world.manager.contexts[TRANSFER_ID] = _context(workflow_run_id=999)
    elif scenario == "workflow_not_found":
        world.db.run.workflow_id = 404
    elif scenario == "provider_cannot_be_built":
        world.provider_error = ValueError("config parked")
    elif scenario == "run_belongs_to_another_provider":
        # Cloudonix's verifier compares x-cx-apikey with the bearer token, and this request sends the matching
        # key: only the provider-name check can refuse it.
        world.provider = _cloudonix()
    before = world.snapshot()

    response = _post_transfer_result(client, signed=True, headers={"x-cx-apikey": FAKE})

    _assert_b028(response)
    assert world.snapshot() == before


@pytest.mark.parametrize("carrier", ["twilio", "telnyx"])
def test_signed_callback_of_a_finished_transfer_is_401_logged_as_info_not_refused(
    client, world, log_lines, carrier
):
    # The strategy removes the context once the transfer completes; the carrier's later 'completed' or
    # 'call.hangup' is genuine. It keeps the 401, but it is not an attack and is not logged as one.
    world.manager.contexts.clear()
    before = world.snapshot()
    if carrier == "twilio":
        response = _post_transfer_result(
            client, signed=True, form={"CallStatus": "completed"}
        )
    else:
        world.provider = _telnyx()
        body = _telnyx_body()
        response = _post_telnyx(client, body, _telnyx_headers(body))

    _assert_b028(response, code="carrier_callback_unauthenticated")
    assert world.snapshot() == before
    gone = [line for line in log_lines if "code=transfer_context_gone" in line]
    assert len(gone) == 1 and "| INFO" in gone[0], log_lines
    assert not [line for line in log_lines if "carrier callback refused" in line]
    assert not [line for line in log_lines if "| WARNING" in line]


def test_transfer_result_signed_answer_publishes_destination_answered(client, world):
    response = _post_transfer_result(client, signed=True)

    assert response.status_code == 200
    assert [e.type for e in world.manager.published] == [
        TransferEventType.DESTINATION_ANSWERED
    ]
    assert world.manager.published[0].original_call_sid == "CA-original"


def test_transfer_result_signed_no_answer_publishes_transfer_failed(client, world):
    form = {"CallStatus": "no-answer", "CallSid": "CA-transfer"}

    response = _post_transfer_result(client, signed=True, form=form)

    assert response.status_code == 200
    assert [e.type for e in world.manager.published] == [
        TransferEventType.TRANSFER_FAILED
    ]


# ---------------------------------------------------------------------------------------------------------
# Telnyx transfer-result
# ---------------------------------------------------------------------------------------------------------


def _telnyx_headers(body: str, *, key=TELNYX_KEY, timestamp=None) -> dict:
    timestamp = timestamp or str(int(time.time()))
    signature = key.sign(f"{timestamp}|{body}".encode()).signature
    return {
        "telnyx-signature-ed25519": base64.b64encode(signature).decode("ascii"),
        "telnyx-timestamp": timestamp,
        "content-type": "application/json",
    }


@pytest.mark.parametrize("headers", ["none", "other_key", "stale", "body_tampered"])
def test_telnyx_transfer_result_bad_signature_is_401_and_moves_nothing(
    client, world, headers
):
    world.provider = _telnyx()
    body = _telnyx_body()
    if headers == "none":
        sent = {"content-type": "application/json"}
    elif headers == "other_key":
        sent = _telnyx_headers(body, key=nacl.signing.SigningKey.generate())
    elif headers == "stale":
        sent = _telnyx_headers(body, timestamp=str(int(time.time()) - 3600))
    else:
        sent = _telnyx_headers(_telnyx_body(cause="no_answer"))
    before = world.snapshot()

    response = _post_telnyx(client, body, sent)

    _assert_b028(response)
    assert world.snapshot() == before


def test_telnyx_transfer_result_without_context_is_401_even_if_signed(client, world):
    world.provider = _telnyx()
    world.manager.contexts.clear()
    body = _telnyx_body()

    response = _post_telnyx(client, body, _telnyx_headers(body))

    _assert_b028(response)
    assert world.manager.published == []


def test_telnyx_transfer_result_signed_hangup_publishes_transfer_failed(client, world):
    world.provider = _telnyx()
    body = _telnyx_body(cause="busy")

    response = _post_telnyx(client, body, _telnyx_headers(body))

    assert response.status_code == 200
    assert [e.type for e in world.manager.published] == [
        TransferEventType.TRANSFER_FAILED
    ]
    assert world.manager.published[0].reason == "busy"
