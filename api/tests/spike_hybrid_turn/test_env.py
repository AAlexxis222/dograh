"""Environment smoke test for the 8a spike: pipecat mocks importable, key gate visible."""

import os

import pytest
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.tests import ContextCapturingMockLLM, MockTTSService  # noqa: F401
from pipecat.tests.mock_transport import MockTransport  # noqa: F401


def test_flux_declares_no_ttfs():
    # Spec §1.1: Flux announces ttfs_p99_latency=0.0 because supports_ttfs is False.
    assert DeepgramFluxSTTService.supports_ttfs.fget(object.__new__(DeepgramFluxSTTService)) is False


@pytest.mark.skipif(
    not os.getenv("DEEPGRAM_API_KEY"), reason="gate: DEEPGRAM_API_KEY not set (spec §9)"
)
def test_deepgram_key_present():
    assert os.environ["DEEPGRAM_API_KEY"]
