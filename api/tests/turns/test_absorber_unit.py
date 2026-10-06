"""Absorber rules with direct frames (spec §6.1.1 ratified form, on the Pipecat 1.12 contract):
discard Flux's turn proposals (``Proposed*``), promote the last interim on the local VAD stop
only with a local turn open, emit token deltas, reset per Flux StartOfTurn proposal, discard
the proposals under mute too."""

import asyncio

import pytest
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
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from api.services.pipecat.turns.absorber import TurnSignalAbsorberProcessor
from api.services.pipecat.turns.frames import TranscriptionReplaceFrame


class Recorder(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.down: list[tuple[str, str, bool | None, str]] = []
        self.frames: list[Frame] = []  # the same frames, as they left the absorber

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame,
            (
                TranscriptionFrame,
                InterimTranscriptionFrame,
                TranscriptionReplaceFrame,
                EagerTranscriptionFrame,
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
                ProposedUserStartedSpeakingFrame,
                ProposedUserStoppedSpeakingFrame,
            ),
        ):
            self.frames.append(frame)
            self.down.append(
                (
                    frame.__class__.__name__,
                    getattr(frame, "text", ""),
                    getattr(frame, "finalized", None),
                    getattr(frame, "user_id", ""),
                )
            )
        await self.push_frame(frame, direction)


class Driver(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def run_steps(steps, *, wait_ms=0, hold_ms=1500):
    """``steps``: ("down", frame) = from the STT side; ("up", frame) = the aggregator's
    upstream broadcast; (seconds, None) = sleep."""
    driver, absorber, rec = (
        Driver(),
        TurnSignalAbsorberProcessor(wait_ms=wait_ms, hold_ms=hold_ms),
        Recorder(),
    )
    worker = PipelineWorker(Pipeline([driver, absorber, rec]))

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
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await asyncio.wait_for(asyncio.gather(runner.run(), script()), timeout=15)
    # The absorber's timers live in the TaskManager and every exit cancels or resolves
    # them: an outliving one would push a transcript into the next call's pipeline.
    assert not [
        t
        for t in asyncio.all_tasks()
        if "hybrid-" in (t.get_name() or "") and not t.done()
    ]
    return absorber, rec


def tf(text, finalized=True, user_id="flux"):
    return TranscriptionFrame(text, user_id, "t", finalized=finalized)


def itf(text, user_id="flux"):
    return InterimTranscriptionFrame(text, user_id, "t")


def finals(rec):
    return [
        d
        for d in rec.down
        if d[0]
        in (
            "TranscriptionFrame",
            "HeldTranscriptionFrame",
            "PromotedTranscriptionFrame",
            "TranscriptionReplaceFrame",
        )
    ]


@pytest.mark.asyncio
async def test_discards_flux_proposals_and_promotes_on_local_vad_stop():
    absorber, rec = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),  # Flux StartOfTurn
            ("up", UserStartedSpeakingFrame()),  # local turn opened
            ("down", itf("hola quiero")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
        ]
    )
    assert not any(d[0] == "ProposedUserStartedSpeakingFrame" for d in rec.down)
    assert (
        "PromotedTranscriptionFrame",
        "hola quiero",
        False,
        "flux",
    ) in rec.down  # §5.3-11 user_id kept
    assert absorber.stats["promoted"] == 1


@pytest.mark.asyncio
async def test_vad_stop_without_local_turn_open_does_not_promote():
    absorber, rec = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),
            ("down", itf("hola quiero")),
            (
                "up",
                VADUserStoppedSpeakingFrame(stop_secs=0.2),
            ),  # no local turn ever opened
        ]
    )
    assert finals(rec) == []
    assert absorber.stats["promoted"] == 0


@pytest.mark.asyncio
async def test_final_extending_promoted_emits_token_delta():
    absorber, rec = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("hola quiero")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            (
                "down",
                tf("Hola, quiero reservar."),
            ),  # capitalised + punctuated: still an extension
            ("down", ProposedUserStoppedSpeakingFrame()),
        ]
    )
    assert finals(rec) == [
        ("PromotedTranscriptionFrame", "hola quiero", False, "flux"),
        ("TranscriptionFrame", "reservar.", True, "flux"),
    ]
    assert absorber.stats["delta_emitted"] == 1


@pytest.mark.asyncio
async def test_final_equal_to_promoted_is_deduped():
    absorber, rec = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("hola quiero")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("down", tf("Hola quiero.")),
            ("down", ProposedUserStoppedSpeakingFrame()),
        ]
    )
    assert len(finals(rec)) == 1
    assert absorber.stats["dup_avoided"] == 1


@pytest.mark.asyncio
async def test_blank_final_after_a_promotion_adds_nothing():
    # A blank final has fewer tokens than the promotion, which ``token_delta`` reads as a
    # rewrite: it must be dropped before it reaches the replace branch.
    absorber, rec = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("hola quiero")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("down", tf("   ")),
            ("down", ProposedUserStoppedSpeakingFrame()),
        ]
    )
    assert [d[1] for d in finals(rec)] == ["hola quiero"]
    assert absorber.stats["dup_avoided"] == 1


@pytest.mark.asyncio
async def test_second_local_turn_in_same_flux_turn_promotes_only_the_delta():
    absorber, rec = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("quiero reservar")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("up", UserStoppedSpeakingFrame()),
            (
                "up",
                UserStartedSpeakingFrame(),
            ),  # second local turn, same Flux turn (S5)
            ("down", itf("quiero reservar para el sábado")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
        ]
    )
    assert [d[1] for d in finals(rec)] == ["quiero reservar", "para el sábado"]
    assert absorber.stats["promoted"] == 2


@pytest.mark.asyncio
async def test_final_with_nothing_emitted_and_local_turn_open_passes_through():
    absorber, rec = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", tf("hola")),  # no interim (S4)
            ("down", ProposedUserStoppedSpeakingFrame()),
        ]
    )
    assert finals(rec) == [("TranscriptionFrame", "hola", True, "flux")]
    assert absorber.stats["passthrough_final"] == 1


@pytest.mark.asyncio
async def test_start_of_turn_resets_stale_state_from_a_turn_closed_without_final():
    # §5.3-13: Flux dropped the final (min_confidence); its interim must not be promoted
    # into the next turn and the next final must not be treated as an orphan.
    absorber, rec = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("stale words")),
            ("down", ProposedUserStoppedSpeakingFrame()),  # Flux closed, no final
            ("up", UserStoppedSpeakingFrame()),
            ("down", ProposedUserStartedSpeakingFrame()),  # next Flux turn → reset here
            ("up", UserStartedSpeakingFrame()),
            (
                "up",
                VADUserStoppedSpeakingFrame(stop_secs=0.2),
            ),  # nothing to promote yet
            ("down", tf("fresh words")),
            ("down", ProposedUserStoppedSpeakingFrame()),
        ]
    )
    assert [d[1] for d in finals(rec)] == ["fresh words"]
    assert absorber.stats["turn_closed_without_final"] == 1


@pytest.mark.asyncio
async def test_muted_proposals_are_discarded_and_nothing_is_promoted():
    absorber, rec = await run_steps(
        [
            ("up", UserMuteStartedFrame()),
            ("down", ProposedUserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("while muted")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("down", ProposedUserStoppedSpeakingFrame()),
            ("up", UserMuteStoppedFrame()),
        ]
    )
    # 1.7 passed the signals through under mute (§5.3-4); a 1.12 proposal that crosses would
    # leave the aggregator's ``_user_speaking`` set (spec B2 VOZ-AC-B2-12), so none does.
    assert not any(d[0].startswith("Proposed") for d in rec.down)
    assert absorber.stats["promoted"] == 0


@pytest.mark.asyncio
async def test_interruption_reemits_last_interim():
    absorber, rec = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),
            ("down", itf("hola quiero")),
            ("up", InterruptionFrame()),
        ]
    )
    interims = [d for d in rec.down if d[0] == "InterimTranscriptionFrame"]
    assert len(interims) == 2
    assert absorber.stats["reemitted_interim"] == 1


@pytest.mark.asyncio
async def test_interruption_after_a_final_does_not_reemit_the_consumed_interim():
    # The turn outlives its final (§5.3-9), so the interim it superseded is still there:
    # re-arming the start strategies with it would open a ghost turn.
    absorber, _ = await run_steps(
        [
            ("down", ProposedUserStartedSpeakingFrame()),
            ("up", UserStartedSpeakingFrame()),
            ("down", itf("hola quiero")),
            ("up", VADUserStoppedSpeakingFrame(stop_secs=0.2)),
            ("down", tf("hola quiero reservar")),
            ("up", InterruptionFrame()),
        ]
    )
    assert absorber.stats["reemitted_interim"] == 0
