"""Candidate TurnSignalAbsorberProcessor forms F1/F2 (spec §3). Throwaway. Sits right after
the STT. Tracks state per Flux turn using the aggregator's UPSTREAM broadcasts (spec §1.2)."""
import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from pipecat.frames.frames import (
    CancelFrame,
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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601


@dataclass
class FluxTurn:
    last_interim: str | None = None
    emitted: str | None = None  # text already sent downstream as TranscriptionFrame(s)
    final_seen: bool = False
    local_turns_closed: int = 0
    local_turns_opened: int = 0
    interim_frame: InterimTranscriptionFrame | None = field(default=None, repr=False)


class Absorber(FrameProcessor):
    def __init__(self, *, mode: Literal["f1", "f2"], hybrid_wait_ms: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._mode = mode
        self._wait_s = hybrid_wait_ms / 1000.0
        self._turn: FluxTurn | None = None
        self._local_open = False
        self._muted = False
        self._wait_task: asyncio.Task | None = None
        self.stats: Counter = Counter()

    # ---- helpers -------------------------------------------------------
    def _turn_or_new(self) -> FluxTurn:
        if self._turn is None:
            self._turn = FluxTurn()
        return self._turn

    def _cancel_wait(self) -> None:
        if self._wait_task and not self._wait_task.done():
            self._wait_task.cancel()
        self._wait_task = None

    async def _forward(self, frame: TranscriptionFrame, direction: FrameDirection) -> None:
        """Single exit for transcripts. ``forwarded`` is the harness' loss detector, so every
        TranscriptionFrame that leaves toward the aggregator must be counted exactly once."""
        self.stats["forwarded"] += 1
        await self.push_frame(frame, direction)

    async def _promote(self) -> None:
        turn = self._turn
        if turn is None or turn.final_seen or self._muted or not turn.last_interim:
            return
        if turn.emitted == turn.last_interim:
            return
        turn.emitted = turn.last_interim
        self.stats["promoted"] += 1
        await self._forward(
            TranscriptionFrame(turn.last_interim, "", time_now_iso8601(), finalized=False),
            FrameDirection.DOWNSTREAM,
        )

    async def _wait_then_promote(self) -> None:
        try:
            await asyncio.sleep(self._wait_s)
        except asyncio.CancelledError:
            return
        self.stats["wait_expired"] += 1
        await self._promote()

    async def _on_local_vad_stop(self) -> None:
        if self._muted or self._turn is None or self._turn.final_seen:
            return
        if self._mode == "f1":
            await self._promote()
        else:
            self._cancel_wait()
            self._wait_task = asyncio.create_task(self._wait_then_promote())

    async def _on_flux_final(self, frame: TranscriptionFrame, direction: FrameDirection) -> None:
        turn = self._turn_or_new()
        self._cancel_wait()
        turn.final_seen = True
        emitted = turn.emitted
        text = frame.text
        try:
            if emitted is None:
                self.stats["passthrough_final"] += 1
                await self._forward(frame, direction)
                return
            # ``None`` marks a rewrite: the final does not extend what we already emitted.
            delta = text[len(emitted):].strip() if text.startswith(emitted) else None
            if not self._local_open:
                # Orphan rule (spec §3): text already went out for this Flux turn and no local
                # turn is open, so anything pushed here would open a ghost turn. Takes priority
                # over the delta/dup/rewrite branches below.
                self.stats["orphan_avoided"] += 1
                if delta is None or delta:  # a rewrite, or text beyond what we emitted
                    self.stats["orphan_text_dropped"] += 1
                return
            # Reaching here implies a local turn is open, which is why there is no
            # ``rewrite_dropped`` branch: the orphan rule above already covers local-closed.
            if delta is None:
                self.stats["rewrite"] += 1
                await self._forward(frame, direction)
            elif delta:
                self.stats["delta_emitted"] += 1
                await self._forward(
                    TranscriptionFrame(delta, frame.user_id, frame.timestamp, finalized=True),
                    direction,
                )
            else:
                self.stats["dup_avoided"] += 1
        finally:
            self._turn = None

    # ---- frame routing -------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM:
            # Aggregator-originated signals (agg:1186-1199, :1236, :1241, :1290, :1112-1115).
            if isinstance(frame, UserStartedSpeakingFrame):
                self._local_open = True
                self._turn_or_new().local_turns_opened += 1
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._local_open = False
                self._cancel_wait()
                if self._turn is not None:
                    self._turn.local_turns_closed += 1
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                await self._on_local_vad_stop()
            elif isinstance(frame, UserMuteStartedFrame):
                self._muted = True
                self._cancel_wait()
            elif isinstance(frame, UserMuteStoppedFrame):
                self._muted = False
            elif isinstance(frame, InterruptionFrame):
                # Queues were flushed (fp:871-890); interims are idempotent for the aggregator
                # (agg:797), so re-send the last one to re-arm start strategies.
                if self._turn is not None and self._turn.interim_frame is not None:
                    self.stats["reemitted_interim"] += 1
                    f = self._turn.interim_frame
                    await self.push_frame(InterimTranscriptionFrame(f.text, f.user_id, f.timestamp))
            await self.push_frame(frame, direction)
            return

        # DOWNSTREAM: STT-originated frames.
        if isinstance(frame, UserStartedSpeakingFrame):
            self._turn_or_new()
            self.stats["swallowed_UserStartedSpeakingFrame"] += 1
            return
        if isinstance(frame, UserStoppedSpeakingFrame):
            self.stats["swallowed_UserStoppedSpeakingFrame"] += 1  # never a barrier (spec §1.1)
            return
        if isinstance(frame, InterimTranscriptionFrame):
            turn = self._turn_or_new()
            turn.last_interim = frame.text
            turn.interim_frame = frame
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TranscriptionFrame):
            await self._on_flux_final(frame, direction)
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            self._cancel_wait()
            self._turn = None
        await self.push_frame(frame, direction)
