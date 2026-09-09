"""Absorber rules with direct frames (spec §3): swallow Flux turn signals, promote on local
VAD-stop, delta/rewrite/orphan handling of Flux's final, reset table."""
import asyncio

import pytest
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from api.tests.spike_hybrid_turn.absorber import Absorber


class Recorder(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.down: list[tuple[str, str, bool | None]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (TranscriptionFrame, InterimTranscriptionFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame)
        ):
            self.down.append((frame.__class__.__name__, getattr(frame, "text", ""), getattr(frame, "finalized", None)))
        await self.push_frame(frame, direction)


class Driver(FrameProcessor):
    """Sits before the absorber; pushes scripted frames downstream. Upstream frames come from
    the recorder (simulating the aggregator's upstream broadcasts)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def _run(mode, steps, wait_ms=0):
    driver, absorber, rec = Driver(), Absorber(mode=mode, hybrid_wait_ms=wait_ms), Recorder()
    task = PipelineTask(Pipeline([driver, absorber, rec]))

    async def script():
        await asyncio.sleep(0.05)
        for target, frame in steps:
            if target == "down":
                await driver.push_frame(frame)
            elif target == "up":
                await rec.push_frame(frame, FrameDirection.UPSTREAM)
            else:
                await asyncio.sleep(target)
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        await task.queue_frame(EndFrame())

    runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    await asyncio.wait_for(asyncio.gather(script(), runner), timeout=15)
    return absorber, rec


def tf(text, finalized=True):
    return TranscriptionFrame(text, "", "t", finalized=finalized)


@pytest.mark.asyncio
async def test_f1_swallows_flux_signals_and_promotes_on_vad_stop():
    absorber, rec = await _run("f1", [
        ("down", UserStartedSpeakingFrame()),
        ("up", UserStartedSpeakingFrame()),  # local turn opened
        ("down", InterimTranscriptionFrame("hola quiero", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
    ])
    assert ("UserStartedSpeakingFrame", "", None) not in rec.down
    assert ("TranscriptionFrame", "hola quiero", False) in rec.down
    assert absorber.stats["promoted"] == 1


@pytest.mark.asyncio
async def test_f1_final_extending_promoted_emits_delta_while_turn_open():
    absorber, rec = await _run("f1", [
        ("down", UserStartedSpeakingFrame()),
        ("up", UserStartedSpeakingFrame()),
        ("down", InterimTranscriptionFrame("hola quiero", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
        ("down", tf("hola quiero reservar")),
        ("down", UserStoppedSpeakingFrame()),
    ])
    finals = [d for d in rec.down if d[0] == "TranscriptionFrame"]
    assert finals == [("TranscriptionFrame", "hola quiero", False), ("TranscriptionFrame", "reservar", True)]
    assert absorber.stats["forwarded"] == 2  # promoted interim + delta (harness loss detector)


@pytest.mark.asyncio
async def test_f1_vad_stop_without_local_turn_open_does_not_promote():
    """Spec §3: promotion needs a local turn OPEN; otherwise it would open a ghost turn."""
    absorber, rec = await _run("f1", [
        ("down", UserStartedSpeakingFrame()),
        ("up", UserStartedSpeakingFrame()),
        ("down", InterimTranscriptionFrame("hola", "", "t")),
        ("up", UserStoppedSpeakingFrame()),  # local turn closed, Flux turn still open
        ("down", InterimTranscriptionFrame("hola quiero", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
    ])
    assert absorber.stats["promoted"] == 0
    assert not [d for d in rec.down if d[0] == "TranscriptionFrame"]


@pytest.mark.asyncio
async def test_f1_second_local_turn_promotes_only_the_delta():
    """A growing interim across two local turns of one Flux turn must not re-send emitted text."""
    absorber, rec = await _run("f1", [
        ("down", UserStartedSpeakingFrame()),
        ("up", UserStartedSpeakingFrame()),
        ("down", InterimTranscriptionFrame("hola quiero", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
        ("up", UserStoppedSpeakingFrame()),
        ("up", UserStartedSpeakingFrame()),  # second local turn, same Flux turn
        ("down", InterimTranscriptionFrame("hola quiero reservar", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
    ])
    finals = [d for d in rec.down if d[0] == "TranscriptionFrame"]
    assert finals == [("TranscriptionFrame", "hola quiero", False), ("TranscriptionFrame", "reservar", False)]
    assert absorber.stats["promoted"] == 2
    assert absorber.stats["forwarded"] == 2


@pytest.mark.asyncio
async def test_f1_orphan_final_after_local_close_is_swallowed():
    absorber, rec = await _run("f1", [
        ("down", UserStartedSpeakingFrame()),
        ("up", UserStartedSpeakingFrame()),
        ("down", InterimTranscriptionFrame("hola quiero", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
        ("up", UserStoppedSpeakingFrame()),  # local turn closed
        ("down", tf("hola quiero")),
        ("down", UserStoppedSpeakingFrame()),
    ])
    assert [d for d in rec.down if d[0] == "TranscriptionFrame"] == [("TranscriptionFrame", "hola quiero", False)]
    assert absorber.stats["orphan_avoided"] == 1


@pytest.mark.asyncio
async def test_rewrite_final_passes_whole_and_is_counted():
    absorber, rec = await _run("f1", [
        ("down", UserStartedSpeakingFrame()),
        ("up", UserStartedSpeakingFrame()),
        ("down", InterimTranscriptionFrame("ola quiero", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
        ("down", tf("hola quiero reservar")),
    ])
    assert ("TranscriptionFrame", "hola quiero reservar", True) in rec.down
    assert absorber.stats["rewrite"] == 1


@pytest.mark.asyncio
async def test_f2_waits_then_promotes_on_expiry():
    absorber, rec = await _run("f2", [
        ("down", UserStartedSpeakingFrame()),
        ("up", UserStartedSpeakingFrame()),
        ("down", InterimTranscriptionFrame("hola quiero", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
        (0.5, None),  # wait beyond hybrid_wait_ms=300
    ], wait_ms=300)
    assert ("TranscriptionFrame", "hola quiero", False) in rec.down
    assert absorber.stats["wait_expired"] == 1


@pytest.mark.asyncio
async def test_f2_final_before_expiry_cancels_promotion():
    absorber, rec = await _run("f2", [
        ("down", UserStartedSpeakingFrame()),
        ("up", UserStartedSpeakingFrame()),
        ("down", InterimTranscriptionFrame("hola quiero", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
        ("down", tf("hola quiero reservar")),
        (0.5, None),
    ], wait_ms=300)
    finals = [d for d in rec.down if d[0] == "TranscriptionFrame"]
    assert finals == [("TranscriptionFrame", "hola quiero reservar", True)]
    assert absorber.stats["promoted"] == 0
    assert absorber.stats["forwarded"] == 1  # passthrough final only


@pytest.mark.asyncio
async def test_mute_disables_promotion_until_unmuted():
    absorber, rec = await _run("f1", [
        ("up", UserMuteStartedFrame()),
        ("down", UserStartedSpeakingFrame()),
        ("down", InterimTranscriptionFrame("hola", "", "t")),
        ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
        ("up", UserMuteStoppedFrame()),
    ])
    assert absorber.stats["promoted"] == 0
    assert not [d for d in rec.down if d[0] == "TranscriptionFrame"]
    # Reset table (spec §3): under mute the absorber stops swallowing Flux's turn signals too.
    assert ("UserStartedSpeakingFrame", "", None) in rec.down
    assert absorber.stats["passthrough_muted_signal"] == 1
    assert absorber.stats["swallowed_UserStartedSpeakingFrame"] == 0


@pytest.mark.asyncio
async def test_interruption_reemits_last_interim():
    absorber, rec = await _run("f1", [
        ("down", UserStartedSpeakingFrame()),
        ("down", InterimTranscriptionFrame("hola quiero", "", "t")),
        ("up", InterruptionFrame()),
    ])
    assert absorber.stats["reemitted_interim"] == 1
    assert rec.down.count(("InterimTranscriptionFrame", "hola quiero", None)) == 2
