"""S10: a consumer observes Flux's UserStoppedSpeakingFrame BEFORE the interim and the final
(spec §1.1, measured 2026-09-09). The stub must reproduce the service's call pattern so the
real pipecat queues yield that order."""

import asyncio

import pytest
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from api.tests.spike_hybrid_turn.flux_stub import FluxStub
from api.tests.spike_hybrid_turn.timeline import Timeline


class Sink(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame,
            (InterimTranscriptionFrame, TranscriptionFrame, UserStoppedSpeakingFrame),
        ):
            self.seen.append(frame.__class__.__name__)
        await self.push_frame(frame, direction)


async def _run(yield_between: bool) -> list[str]:
    timeline = Timeline()
    stub = FluxStub(timeline)
    sink = Sink()
    task = PipelineTask(Pipeline([stub, sink]))

    async def script():
        await stub.emit_start_of_turn()
        await stub.emit_eager("hola quiero")
        await stub.emit_end_of_turn("hola quiero reservar", yield_between=yield_between)

    timeline.at(50, script)
    timeline.at(400, lambda: task.queue_frame(EndFrame()))
    runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    await asyncio.wait_for(asyncio.gather(timeline.run(), runner), timeout=15)
    return sink.seen


@pytest.mark.asyncio
async def test_user_stopped_overtakes_transcripts():
    seen = await _run(yield_between=False)
    # Invariant (spec §1.1): the final is always overtaken by UserStopped. Where the interim
    # lands depends on the Eager->End gap (burst here => UserStopped first; measured 2026-09-09).
    assert seen.index("UserStoppedSpeakingFrame") < seen.index("TranscriptionFrame"), (
        seen
    )
    assert seen == [
        "UserStoppedSpeakingFrame",
        "InterimTranscriptionFrame",
        "TranscriptionFrame",
    ], seen


@pytest.mark.asyncio
async def test_mutation_yielding_between_changes_order():
    # Mutation control: if the stub yields the loop between final and UserStopped, the
    # observed order is no longer the production one (spec §4 mutation rule for S10).
    seen = await _run(yield_between=True)
    assert seen.index("TranscriptionFrame") < seen.index("UserStoppedSpeakingFrame"), (
        seen
    )
