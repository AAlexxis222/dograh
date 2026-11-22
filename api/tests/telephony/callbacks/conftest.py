"""Fixtures of the carrier-callback tests: the fakes of api.tests.support.carrier_callbacks patched in, the app and the log sink."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from api.routes.telephony import router
from api.tests.support.carrier_callbacks import World


@pytest.fixture
def world():
    w = World()
    get_manager = AsyncMock(return_value=w.manager)
    with (
        patch("api.services.telephony.callback_auth.db_client", w.db),
        patch(
            "api.services.telephony.callback_auth.get_call_transfer_manager",
            get_manager,
        ),
        patch("api.routes.telephony.get_call_transfer_manager", get_manager),
        patch(
            "api.services.telephony.providers.telnyx.routes.get_call_transfer_manager",
            get_manager,
        ),
        patch(
            "api.services.telephony.callback_auth.get_telephony_provider_for_run",
            side_effect=w.provider_for_run,
        ),
        patch("api.routes.telephony.db_client", w.db),
        patch("api.services.telephony.providers.cloudonix.routes.db_client", w.db),
        patch(
            "api.services.telephony.providers.cloudonix.routes.get_call_transfer_manager",
            get_manager,
        ),
        patch(
            "api.services.telephony.providers.cloudonix.routes._process_status_update",
            side_effect=lambda run_id, update: w.status_updates.append(run_id),
        ),
        patch("api.services.telephony.providers.vonage.routes.db_client", w.db),
        patch(
            "api.services.telephony.providers.vonage.routes."
            "get_telephony_provider_for_run",
            side_effect=w.provider_for_run,
        ),
    ):
        yield w


@pytest.fixture
def client(world):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def log_lines():
    lines = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    try:
        yield lines
    finally:
        logger.remove(sink_id)
