"""Real pipecat classes for the aggregator seam tests (no mocks of the aggregator or the
greeting controller)."""

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext

from api.services.pipecat.greeting import GreetingController
from api.services.pipecat.speech_playback import SpeechPlaybackTracker


@pytest.fixture
def llm_context():
    return LLMContext(messages=[{"role": "system", "content": "t"}])


@pytest.fixture
def playback():
    return SpeechPlaybackTracker()


@pytest.fixture
def greeting_controller(playback, llm_context):
    return GreetingController(playback, lambda: llm_context, is_screening=lambda: False)
