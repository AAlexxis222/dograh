"""The ARI answer to a refused admission. The HTTP, WebSocket, signaling and carrier answers are proven through
their routes: test_telephony_routes, test_public_agent_routes, test_agent_stream_route and
test_webrtc_signaling_concurrency."""

import pytest

from api.routes.admission_answers import ari_hangup_note
from api.services.call_concurrency import (
    AdmissionBackendUnavailableError,
    CallConcurrencyLimitError,
)


@pytest.mark.parametrize(
    ("refusal", "code"),
    [
        (CallConcurrencyLimitError, "concurrent_call_limit"),
        (AdmissionBackendUnavailableError, "admission_backend_unavailable"),
    ],
)
def test_ari_hangup_names_the_true_reason(refusal, code):
    error = refusal(
        organization_id=11, source="ari_inbound", wait_time=0, max_concurrent=2
    )
    assert ari_hangup_note(error, 11, "chan-1") == (
        f"[ARI org=11] Call admission refused ({code}); "
        "hanging up inbound channel chan-1"
    )
