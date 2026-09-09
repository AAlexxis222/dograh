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
        if mode not in ("f1", "f2"):
            # A typo would silently run F2 and be labelled F1 in the results table.
            raise ValueError(f"Absorber mode must be 'f1' or 'f2', got {mode!r}")
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

    @staticmethod
    def _delta_beyond(emitted: str, text: str) -> str | None:
        """What ``text`` adds beyond the already-emitted ``emitted``.

        ``""`` means it adds nothing; ``None`` marks a rewrite (``text`` does not extend
        ``emitted``). Shared by the promotion path and the Flux-final path so both read the
        same relationship the same way.
        """
        return text[len(emitted):].strip() if text.startswith(emitted) else None

    async def _promote(self) -> None:
        turn = self._turn
        if turn is None or turn.final_seen or self._muted or not turn.last_interim:
            return
        text = turn.last_interim
        if turn.emitted is None:
            payload = text
        else:
            # A second local turn inside the same Flux turn sees a longer interim; sending it
            # whole would re-deliver what turn 1 already got (spec §3 B9).
            delta = self._delta_beyond(turn.emitted, text)
            if delta == "":
                return
            if delta is None:
                self.stats["rewrite"] += 1  # not an extension: re-send whole, same as the final
                payload = text
            else:
                payload = delta
        turn.emitted = text
        self.stats["promoted"] += 1
        await self._forward(
            TranscriptionFrame(payload, "", time_now_iso8601(), finalized=False),
            FrameDirection.DOWNSTREAM,
        )

    async def _swallow_turn_signal(self, frame: Frame, direction: FrameDirection, key: str) -> None:
        """Swallow one of Flux's downstream turn signals — unless muted.

        Reset table (spec §3): under mute the absorber stops promoting AND stops swallowing.
        The aggregator suppresses these signals itself while muted (agg:1085-1094), so letting
        them through keeps the mute path observable instead of hiding it here.
        """
        if self._muted:
            self.stats["passthrough_muted_signal"] += 1
            await self.push_frame(frame, direction)
            return
        self.stats[key] += 1

    async def _wait_then_promote(self) -> None:
        try:
            await asyncio.sleep(self._wait_s)
        except asyncio.CancelledError:
            return
        self.stats["wait_expired"] += 1
        await self._promote()

    async def _on_local_vad_stop(self) -> None:
        # Spec §3: promote (F1) or arm the timer (F2) only with a local turn OPEN and no Flux
        # final for this turn. Without the open-turn check a VAD stop after the local turn
        # closed would push text into no turn at all, i.e. open a ghost turn.
        if self._muted or not self._local_open or self._turn is None or self._turn.final_seen:
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
            delta = self._delta_beyond(emitted, text)
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
            await self._swallow_turn_signal(frame, direction, "swallowed_UserStartedSpeakingFrame")
            return
        if isinstance(frame, UserStoppedSpeakingFrame):  # never a barrier (spec §1.1)
            await self._swallow_turn_signal(frame, direction, "swallowed_UserStoppedSpeakingFrame")
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
