"""VOZ-OLA0-F6 regression cases (fork PRs #10/#15) ported to the upstream
SpeechPlaybackTracker / queue_speech API (VOZ-AC-B2-68, VOZ-AT-B2-15).

The mute observable is the engine's verdict for the next user frame
(``should_mute_user``), read through the ``mock_engine_with_worker`` fixture.
"""

import asyncio

import pytest
from pipecat.frames.frames import TTSAudioRawFrame

import api.services.workflow.pipecat_engine as pe
from api.services.pipecat.speech_playback import PlaybackOutcome
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.pipecat_engine_custom_tools import _parse_timeout_s


async def _wait_until_started(speech) -> None:
    async with asyncio.timeout(1):
        while not speech.started:
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_queued_speech_mutes_user_until_playback_finishes(
    mock_engine_with_worker,
):
    engine, worker = mock_engine_with_worker
    await engine.queue_speech("Un momento, por favor.", mute_user=True)
    assert await worker.is_user_muted() is True  # muted while the speech is pending
    await worker.finish_playback_of_last_speech()
    assert (
        await worker.is_user_muted() is False
    )  # released by the tracker, not by a watchdog


@pytest.mark.asyncio
async def test_cancelled_speech_does_not_leak_mute(mock_engine_with_worker):
    engine, worker = mock_engine_with_worker
    await engine.queue_speech("Te paso con reservas.", mute_user=True)
    await worker.cancel_last_speech()  # interrupted / flushed speech
    assert (
        await worker.is_user_muted() is False
    )  # PR #15: a cancelled speech must not keep the hold


def test_no_queued_speech_mute_symbols_remain():
    engine = PipecatEngine(workflow=None, call_context_vars={})
    names = [*dir(pe), *dir(PipecatEngine), *vars(engine)]
    assert not [
        n for n in names if "_queued_speech_mute" in n
    ]  # VOZ-AC-B2-68: holds deleted


@pytest.mark.asyncio
async def test_silent_tts_releases_the_mute_at_its_deadline(mock_engine_with_worker):
    """Speech that never plays must not mute the caller for the whole call.

    A TTS that yields no audio produces no end marker, so the request's own
    deadline is what releases the mute.
    """
    engine, worker = mock_engine_with_worker
    worker.tts.silent = True
    speech = await engine.queue_speech("Hola.", mute_user=True, timeout=0.05)
    assert await worker.is_user_muted() is True

    assert await asyncio.wait_for(speech.wait(), 1) is False

    assert speech.outcome is PlaybackOutcome.TIMED_OUT
    assert await worker.is_user_muted() is False


@pytest.mark.asyncio
async def test_started_playback_keeps_the_mute_until_its_end(mock_engine_with_worker):
    """Once the speech is playing, only its end releases the mute."""
    engine, worker = mock_engine_with_worker
    speech = await engine.queue_speech("Un momento.", mute_user=True)
    await _wait_until_started(speech)
    await asyncio.sleep(0.05)

    assert await worker.is_user_muted() is True

    await worker.finish_playback_of_last_speech()

    assert speech.outcome is PlaybackOutcome.PLAYED
    assert await worker.is_user_muted() is False


@pytest.mark.asyncio
async def test_audio_after_the_mute_was_released_does_not_mute_again(
    mock_engine_with_worker,
):
    """A late audio frame must not mute the caller for an utterance that ended."""
    engine, worker = mock_engine_with_worker
    await engine.queue_speech("Un momento.", mute_user=True)
    await worker.finish_playback_of_last_speech()
    assert await worker.is_user_muted() is False

    await worker.output.queue_frame(
        TTSAudioRawFrame(b"\x00\x00" * 640, sample_rate=16000, num_channels=1)
    )
    assert await worker.worker.flush_pipeline(timeout=1)

    assert await worker.is_user_muted() is False


@pytest.mark.asyncio
async def test_unplayed_speech_timeout_keeps_other_speech_muted(
    mock_engine_with_worker,
):
    """The transfer path queues speech without a mute of its own.

    When that speech never starts, its timeout must not release the mute held
    by a different request.
    """
    engine, worker = mock_engine_with_worker
    await engine.queue_speech("Primero.", mute_user=True)
    behind = await engine.queue_speech("Segundo.", timeout=0.05)

    assert await asyncio.wait_for(behind.wait(), 1) is False

    assert behind.outcome is PlaybackOutcome.TIMED_OUT
    assert await worker.is_user_muted() is True


@pytest.mark.asyncio
async def test_started_speech_timeout_keeps_other_speech_muted(mock_engine_with_worker):
    """Playback started but never finished: the other request's mute stays."""
    engine, worker = mock_engine_with_worker
    unmuted = await engine.queue_speech("Primero.", timeout=0.2)
    await _wait_until_started(unmuted)
    await engine.queue_speech("Segundo.", mute_user=True)

    assert await asyncio.wait_for(unmuted.wait(), 1) is False

    assert unmuted.outcome is PlaybackOutcome.TIMED_OUT
    assert await worker.is_user_muted() is True


@pytest.mark.parametrize("raw", ["abc", None, 0, -1])
def test_tool_timeout_rejects_invalid(raw):
    with pytest.raises(ValueError):
        _parse_timeout_s(raw)


def test_tool_timeout_accepts_numeric_string():
    assert _parse_timeout_s("2.5") == 2.5
