"""VOZ-N0-29 (VOZ-AC-B4-46b): the answer URLs GET /ncco, /vobiz-xml, /plivo-xml and /twiml: the run decides whose credentials verify them,
never the query.

The fakes and request builders are shared: api.tests.support.carrier_callbacks and this folder's conftest.py.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.telephony import router
from api.tests.support.carrier_callbacks import (
    ANSWER_URLS,
    BASE,
    NCCO,
    NCCO_QUERY,
    NCCO_TOKEN,
    NOT_AN_OBJECT,
    ORG_ID,
    OTHER_TOKEN,
    PLIVO_NONCE,
    PLIVO_TOKEN,
    PUBLIC_BACKEND,
    RUN_ID,
    VOBIZ_QUERY,
    VOBIZ_SIGNED_URL,
    WORKFLOW_ID,
    _answer_url_patches,
    _assert_b028,
    _get_ncco,
    _plivo,
    _post_vobiz_xml,
    _vobiz,
    _vobiz_headers,
    _vonage_jwt,
)

# ---------------------------------------------------------------------------------------------------------
# Vonage GET /ncco (the answer URL: its NCCO carries the media-stream URL and its token)
# ---------------------------------------------------------------------------------------------------------


def test_ncco_signed_returns_the_ncco(client, world):
    response, build = _get_ncco(client, world)

    assert response.status_code == 200
    assert response.json() == NCCO
    build.assert_awaited_once_with(WORKFLOW_ID, ORG_ID, RUN_ID)


@pytest.mark.parametrize("headers", [{}, "other_secret", "not_a_jwt"])
def test_ncco_bad_signature_is_401_without_ncco_or_token(client, world, headers):
    if headers == "other_secret":
        headers = _vonage_jwt(secret=OTHER_TOKEN)
    elif headers == "not_a_jwt":
        headers = {"authorization": "Bearer Zm9yZ2Vk"}
    before = world.snapshot()

    response, build = _get_ncco(client, world, headers=headers)

    _assert_b028(response)
    assert NCCO_TOKEN not in response.text and "websocket" not in response.text
    build.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize(
    "tampered", [{"workflow_id": 34}, {"organization_id": 12}, {"workflow_run_id": 8}]
)
def test_ncco_query_that_disagrees_with_the_run_is_401_even_if_signed(
    client, world, tampered
):
    response, build = _get_ncco(client, world, query={**NCCO_QUERY, **tampered})

    _assert_b028(response)
    assert NCCO_TOKEN not in response.text
    build.assert_not_awaited()


# ---------------------------------------------------------------------------------------------------------
# Vobiz /vobiz-xml (the answer URL: its XML carries the media-stream URL and its token)
# ---------------------------------------------------------------------------------------------------------


def test_vobiz_xml_signed_returns_the_stream_xml(client, world):
    response, build = _post_vobiz_xml(client, world)

    assert response.status_code == 200
    build.assert_awaited_once_with(WORKFLOW_ID, ORG_ID, RUN_ID)


@pytest.mark.parametrize("headers", [{}, "forged", "other_token"])
def test_vobiz_xml_bad_signature_is_401_and_builds_no_stream_url(
    client, world, headers
):
    if headers == "forged":
        headers = {
            "x-vobiz-signature-v3": "Zm9yZ2Vk",
            "x-vobiz-signature-v3-nonce": "12345678901234567890",
        }
    elif headers == "other_token":
        headers = _vobiz_headers(VOBIZ_SIGNED_URL, token=OTHER_TOKEN)
    before = world.snapshot()

    response, build = _post_vobiz_xml(client, world, headers=headers)

    _assert_b028(response)
    build.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize(
    "tampered", [{"workflow_id": 34}, {"organization_id": 12}, {"workflow_run_id": 8}]
)
def test_vobiz_xml_query_that_disagrees_with_the_run_is_401_even_if_signed(
    client, world, tampered
):
    # Vobiz signs the URL without its query string, so a replayed signature covers any query. The run decides.
    response, build = _post_vobiz_xml(client, world, query={**VOBIZ_QUERY, **tampered})

    _assert_b028(response)
    build.assert_not_awaited()


def test_vobiz_xml_verifies_the_public_url_vobiz_signed_not_the_one_the_app_sees(
    client, world
):
    # Behind a TLS proxy or tunnel the app sees http://<internal host>; Vobiz signed the URL the platform gave it.
    with patch(
        "api.services.telephony.providers.vobiz.routes.get_backend_endpoints",
        new=AsyncMock(return_value=(PUBLIC_BACKEND, "wss://public.example")),
    ):
        response, build = _post_vobiz_xml(
            client,
            world,
            headers=_vobiz_headers(f"{PUBLIC_BACKEND}/api/v1/telephony/vobiz-xml"),
        )

    assert response.status_code == 200, response.text
    build.assert_awaited_once_with(WORKFLOW_ID, ORG_ID, RUN_ID)


def test_vobiz_xml_signed_over_the_internal_url_is_401(client, world):
    with patch(
        "api.services.telephony.providers.vobiz.routes.get_backend_endpoints",
        new=AsyncMock(return_value=(PUBLIC_BACKEND, "wss://public.example")),
    ):
        response, build = _post_vobiz_xml(
            client, world, headers=_vobiz_headers(BASE + "/telephony/vobiz-xml")
        )

    _assert_b028(response)
    build.assert_not_awaited()


@pytest.mark.parametrize("body", NOT_AN_OBJECT)
def test_parse_webhook_request_refuses_a_json_body_that_is_not_an_object(body):
    from api.utils.telephony_helper import parse_webhook_request

    class JsonRequest:
        query_params = {}

        async def body(self):
            return body.encode()

        async def json(self):
            return json.loads(body)

        async def form(self):
            raise AssertionError("a JSON body is not read as a form")

    with pytest.raises(ValueError):
        asyncio.run(parse_webhook_request(JsonRequest()))


@pytest.mark.parametrize("body", NOT_AN_OBJECT)
def test_vobiz_xml_with_a_body_that_is_not_an_object_is_401_before_any_lookup(
    client, world, body
):
    world.provider = _vobiz()
    world.provider.get_webhook_response = AsyncMock(return_value="<Response/>")
    lookup = AsyncMock(side_effect=world.db.get_workflow_run_by_id)
    world.db.get_workflow_run_by_id = lookup
    before = world.snapshot()
    with patch(
        "api.services.telephony.providers.vobiz.routes.get_backend_endpoints",
        new=AsyncMock(return_value=(PUBLIC_BACKEND, "wss://public.example")),
    ):
        response = client.post(
            f"/telephony/vobiz-xml?{urlencode(VOBIZ_QUERY)}",
            content=body,
            headers={
                **_vobiz_headers(VOBIZ_SIGNED_URL),
                "content-type": "application/json",
            },
        )

    _assert_b028(response)
    lookup.assert_not_awaited()
    world.provider.get_webhook_response.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize("kind", ["hangup", "ring"])
@pytest.mark.parametrize("body", NOT_AN_OBJECT)
def test_vobiz_status_callback_with_a_body_that_is_not_an_object_is_403(
    world, kind, body
):
    # Each Vobiz callback keeps one refusal status: the 403 it answers for a bad signature (Vobiz docs).
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)
    lookup = AsyncMock(side_effect=world.db.get_workflow_run_by_id)
    world.db.get_workflow_run_by_id = lookup
    before = world.snapshot()
    with patch("api.services.telephony.providers.vobiz.routes.db_client", world.db):
        response = client.post(
            f"/telephony/vobiz/{kind}-callback/{RUN_ID}",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 403, response.text
    lookup.assert_not_awaited()
    assert world.snapshot() == before


# ---------------------------------------------------------------------------------------------------------
# /plivo-xml and /twiml: the answer URL; the run decides whose credentials verify it, never the query
# ---------------------------------------------------------------------------------------------------------


ATTACKER_ORG = 12


@pytest.mark.parametrize("route", sorted(ANSWER_URLS))
def test_answer_url_naming_another_org_is_401_even_if_signed_by_that_org(
    client, world, route
):
    # The caller's own org has a configuration of the same carrier and signs with its own token; the run
    # belongs to ORG_ID. The query's organization_id must not choose the credentials.
    spec = ANSWER_URLS[route]
    own, attacker = spec["provider"](PLIVO_TOKEN), spec["provider"](OTHER_TOKEN)
    providers = {ORG_ID: own, ATTACKER_ORG: attacker}
    for provider in providers.values():
        provider.get_webhook_response = AsyncMock(return_value="<Response/>")

    async def resolve(run, organization_id):
        return providers[organization_id]

    query = {
        "workflow_id": WORKFLOW_ID,
        "workflow_run_id": RUN_ID,
        "organization_id": ATTACKER_ORG,
    }
    path = f"/telephony/{route}?{urlencode(query)}"
    before = world.snapshot()
    db, route_lookup, auth_lookup = _answer_url_patches(spec["module"], world, resolve)
    with db, route_lookup, auth_lookup:
        response = client.post(
            path,
            data=spec["form"],
            headers=spec["sign"](BASE + path, spec["form"], attacker),
        )

    _assert_b028(response)
    assert world.snapshot() == before
    own.get_webhook_response.assert_not_awaited()
    attacker.get_webhook_response.assert_not_awaited()


def test_plivo_xml_bad_signature_is_401_and_writes_nothing(client, world):
    world.provider = _plivo()
    world.provider.get_webhook_response = AsyncMock(return_value="<Response/>")
    query = {
        "workflow_id": WORKFLOW_ID,
        "workflow_run_id": RUN_ID,
        "organization_id": ORG_ID,
    }
    before = world.snapshot()
    db, route_lookup, auth_lookup = _answer_url_patches(
        ANSWER_URLS["plivo-xml"]["module"], world, world.provider_for_run
    )
    with db, route_lookup, auth_lookup:
        response = client.post(
            f"/telephony/plivo-xml?{urlencode(query)}",
            data={"CallUUID": "chosen-call-id"},
            headers={
                "x-plivo-signature-v3": "Zm9yZ2Vk",
                "x-plivo-signature-v3-nonce": PLIVO_NONCE,
            },
        )

    _assert_b028(response)
    assert world.snapshot() == before
    world.provider.get_webhook_response.assert_not_awaited()
