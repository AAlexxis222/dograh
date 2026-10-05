"""Drivers for the G0 point-B tests (VOZ-G0-08): push frames through one processor and record
what leaves it, in order and by identity."""

import asyncio
from collections.abc import Callable

from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from api.tests.turns.test_absorber_unit import run_steps

# Every frame a turn engine reads: the STT's text and proposals, the aggregator's broadcasts.
USER_FRAMES = (
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    EagerTranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    InterruptionFrame,
)


async def run_absorber(frames: list[Frame], *, user_muted: bool = False):
    """Push ``frames`` from the STT side through the absorber; return what left it toward the
    aggregator and the absorber's stats. ``user_muted``: the aggregator broadcast its mute
    first."""
    steps = [("up", UserMuteStartedFrame())] if user_muted else []
    absorber, rec = await run_steps(steps + [("down", f) for f in frames])
    return rec.frames, absorber.stats


class FrameRecorder(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.down: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, USER_FRAMES):
            self.down.append(frame)
        await self.push_frame(frame, direction)


class _Source(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def user_frames_out_of(
    make_processor: Callable[[WorkerRunner], FrameProcessor],
    steps: list[Frame | Callable[[FrameProcessor], None]],
) -> list[Frame]:
    """Run ``steps`` downstream through the processor alone: a frame is pushed, a callable is
    called with the processor (to move it to another state mid-run). Returns the user frames
    that left it, in order."""
    runner = WorkerRunner(handle_sigint=False)
    processor = make_processor(runner)
    source, recorder = _Source(), FrameRecorder()
    worker = PipelineWorker(Pipeline([source, processor, recorder]), enable_rtvi=False)

    async def script():
        await asyncio.sleep(0.05)
        for step in steps:
            if isinstance(step, Frame):
                await source.push_frame(step)
            else:
                step(processor)
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)
        await worker.queue_frame(EndFrame())

    await runner.add_workers(worker)
    await asyncio.wait_for(asyncio.gather(runner.run(), script()), timeout=15)
    return recorder.down
