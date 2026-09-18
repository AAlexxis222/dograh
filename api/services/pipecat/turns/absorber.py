"""``TurnSignalAbsorberProcessor`` — the hybrid turn mode (spec §6.1.1, ratified form F1).

Sits right behind a server-turn STT (Deepgram Flux, Dograh-Flux, Cartesia ink-2) when
``turn.source=local``. The STT transcribes; the local VAD + analyzer decide the turns:

* Flux's own ``UserStarted/StoppedSpeakingFrame`` (its StartOfTurn/EndOfTurn) are swallowed
  so they never reach the aggregator's turn controller.
* On the local VAD stop, with a local turn open, the last interim is *promoted* as a
  ``TranscriptionFrame(finalized=False)`` so the local stop strategy has text to close on.
  A second local turn inside the same Flux turn gets only the token delta.
* Flux's final then takes one of four branches: extends what was promoted → the token
  delta; rewrites it → ``TranscriptionReplaceFrame`` (D-11); arrives with no local turn open
  after text went out → the tail is delivered as a message of its own (D-12); arrives with
  nothing emitted and no local turn open → held ``hold_ms`` for the next local turn, then
  delivered as a message of its own (D-13).
* State is indexed by Flux turn and reset on Flux's *next* StartOfTurn, never on its
  UserStopped, which overtakes its own final (spike 8a, S10).

Measured in the 8a spike (``docs/2026-09-09-8a-hybrid-turn-viability`` in the XPAND voice
repo): without this processor the aggregator sees two sets of turn signals and, with a
pause mid-sentence, either loses the tail or opens a ghost turn.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass, field

from loguru import logger

from api.schemas.turn_configuration import (
    DEFAULT_HYBRID_HOLD_MS,
    DEFAULT_HYBRID_WAIT_MS,
)
from api.services.pipecat.turns.frames import TranscriptionReplaceFrame
from api.services.pipecat.turns.text_delta import token_delta
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
    emitted: str | None = None  # text already sent downstream for this Flux turn
    final_seen: bool = False
    stop_seen: bool = False  # Flux's own UserStopped seen (it overtakes the final)
    interim_frame: InterimTranscriptionFrame | None = field(default=None, repr=False)


class TurnSignalAbsorberProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        wait_ms: int = DEFAULT_HYBRID_WAIT_MS,
        hold_ms: int = DEFAULT_HYBRID_HOLD_MS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._wait_s = wait_ms / 1000.0
        self._hold_s = hold_ms / 1000.0
        self._turn: FluxTurn | None = None
        self._local_open = False
        self._muted = False
        self._wait_task: asyncio.Task | None = None
        self._held: tuple[TranscriptionFrame, asyncio.Task] | None = None
        self.stats: Counter = Counter()

    # ---- helpers -------------------------------------------------------
    def _turn_or_new(self) -> FluxTurn:
        if self._turn is None:
            self._turn = FluxTurn()
        return self._turn

    async def _cancel_wait(self) -> None:
        if self._wait_task is not None:
            await self.cancel_task(self._wait_task)
            self._wait_task = None

    async def _forward(self, frame: Frame, direction: FrameDirection) -> None:
        """Single exit for transcripts toward the aggregator (counted once each)."""
        self.stats["forwarded"] += 1
        await self.push_frame(frame, direction)

    async def _promote(self) -> None:
        turn = self._turn
        # ``.strip()``: an interim of pure whitespace is text to nobody. Promoting it
        # would push a blank transcription into the aggregation and leave ``emitted``
        # holding whitespace, which the next final would then diff against.
        if (
            turn is None
            or turn.final_seen
            or self._muted
            or not (turn.last_interim or "").strip()
        ):
            return
        text = turn.last_interim
        if turn.emitted is None:
            payload = text
        else:
            delta = token_delta(turn.emitted, text)
            if delta == "":
                return
            if delta is None:
                self.stats["promoted_rewrite"] += 1
                payload = text
            else:
                payload = delta
        src = turn.interim_frame
        await self._forward(
            TranscriptionFrame(
                payload,
                src.user_id if src else "",
                time_now_iso8601(),
                finalized=False,
            ),
            FrameDirection.DOWNSTREAM,
        )
        # Only after the push: a final arriving inside the push must not see ``emitted``
        # set for text that never left (spike red team §7).
        turn.emitted = text
        self.stats["promoted"] += 1

    async def _swallow_turn_signal(
        self, frame: Frame, direction: FrameDirection, key: str
    ):
        if self._muted:
            # The aggregator suppresses these itself while muted (agg:1085-1094); passing
            # them through keeps the mute path observable (README 8a §5.3-4).
            self.stats["passthrough_muted_signal"] += 1
            await self.push_frame(frame, direction)
            return
        self.stats[key] += 1

    async def _wait_then_promote(self) -> None:
        await asyncio.sleep(self._wait_s)
        self.stats["wait_expired"] += 1
        await self._promote()

    async def _on_local_vad_stop(self) -> None:
        # Promote only with a local turn OPEN and no Flux final for this turn (§5.3-1):
        # otherwise the text would go into no turn at all, i.e. open a ghost turn.
        if (
            self._muted
            or not self._local_open
            or self._turn is None
            or self._turn.final_seen
        ):
            return
        if self._wait_s == 0:
            await self._promote()
            return
        await self._cancel_wait()
        self._wait_task = self.create_task(
            self._wait_then_promote(), name="hybrid-wait"
        )

    # ---- the final held for the next local turn (D-13) ------------------
    async def _hold(self, frame: TranscriptionFrame, direction: FrameDirection) -> None:
        """Nothing emitted and no local turn open → keep the final for the next local
        turn; deliver it on its own when ``hold_ms`` expires or the pipeline ends."""
        self.stats["final_held_no_local_turn"] += 1
        # Flux's finals for a turn are cumulative, so this one already carries whatever
        # an older held one said: keeping both would duplicate that text.
        await self._take_held()
        if self._hold_s == 0:
            await self._release_held_as_message(frame, direction)
            return

        async def expire():
            await asyncio.sleep(self._hold_s)
            held = self._held
            # Cleared by hand instead of through ``_take_held``: this task must never be
            # the one cancelling itself.
            self._held = None
            if held is not None:
                await self._release_held_as_message(held[0], direction)

        self._held = (frame, self.create_task(expire(), name="hybrid-hold"))

    async def _take_held(self) -> TranscriptionFrame | None:
        """Pop the held final and stop its timer. Never call it from that timer."""
        if self._held is None:
            return None
        frame, task = self._held
        self._held = None
        await self.cancel_task(task)
        return frame

    async def _release_held_as_message(
        self, frame: TranscriptionFrame, direction: FrameDirection
    ) -> None:
        self.stats["held_released_as_message"] += 1
        await self._forward(frame, direction)

    async def _release_held_into_turn(self) -> None:
        """A local turn opened while a final was held: hand the text to that turn."""
        frame = await self._take_held()
        if frame is None:
            return
        self.stats["held_released_into_turn"] += 1
        turn = self._turn_or_new()
        await self._forward(frame, FrameDirection.DOWNSTREAM)
        # After the push, as in ``_promote``: a final arriving inside it must not see
        # ``emitted`` set for text that never left.
        turn.emitted = frame.text

    async def _on_flux_final(
        self, frame: TranscriptionFrame, direction: FrameDirection
    ):
        # The turn is NOT cleared here: it lives on with ``final_seen``/``emitted`` until
        # Flux's next StartOfTurn, so a second final extends it (§5.3-8/9/13).
        self.stats["finals_received"] += 1
        turn = self._turn_or_new()
        await self._cancel_wait()
        if turn.final_seen:
            # §5.3-9: a second final for the same Flux turn extends what already went
            # out instead of opening a new turn with the whole text.
            self.stats["second_final_retained"] += 1
        turn.final_seen = True
        emitted, text = turn.emitted, frame.text
        if not text.strip():
            # A blank final carries no text, so it adds nothing and must never reach
            # ``token_delta``, which reads "fewer tokens than emitted" as a rewrite.
            # Dropping it also keeps a blank out of the replace/orphan/hold branches
            # and out of the aggregator, where it would open a ghost turn.
            # Logged because it shares ``dup_avoided`` with "the final repeated what
            # we already sent": the counter alone cannot tell the two apart.
            logger.debug(f"{self}: blank final dropped, it adds no text")
            self.stats["dup_avoided"] += 1
            return
        if emitted is None:
            if self._local_open:
                self.stats["passthrough_final"] += 1
                await self._forward(frame, direction)
                turn.emitted = text
            else:
                await self._hold(frame, direction)
            return
        delta = token_delta(emitted, text)
        if delta == "":
            self.stats["dup_avoided"] += 1
            return
        if not self._local_open:
            # D-12: the local turn closed before the final, so what the final adds
            # beyond it is real speech nobody received → deliver it as a message of its
            # own rather than drop it. A rewrite goes whole: there is no pending
            # aggregation left to replace.
            if delta is None:
                self.stats["orphan_rewrite_emitted"] += 1
                payload = text
            else:
                self.stats["orphan_emitted"] += 1
                payload = delta
            await self._forward(
                TranscriptionFrame(
                    payload, frame.user_id, frame.timestamp, finalized=True
                ),
                direction,
            )
            turn.emitted = text
            return
        if delta is None:
            # D-11: the promoted interim is still pending in the open local turn →
            # replace it, never concatenate.
            self.stats["rewrite_replaced"] += 1
            await self._forward(
                TranscriptionReplaceFrame(text, frame.user_id, frame.timestamp),
                direction,
            )
            turn.emitted = text
            return
        await self._forward(
            TranscriptionFrame(delta, frame.user_id, frame.timestamp, finalized=True),
            direction,
        )
        self.stats["delta_emitted"] += 1
        turn.emitted = text

    # ---- frame routing -------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM:
            # Aggregator-originated broadcasts (agg:1186-1199, :1236, :1241, :1290, :1112-1115).
            if isinstance(frame, UserStartedSpeakingFrame):
                self._local_open = True
                await self._release_held_into_turn()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._local_open = False
                await self._cancel_wait()
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                await self._on_local_vad_stop()
            elif isinstance(frame, UserMuteStartedFrame):
                self._muted = True
                await self._cancel_wait()
            elif isinstance(frame, UserMuteStoppedFrame):
                self._muted = False
            elif isinstance(frame, InterruptionFrame):
                # Queues were flushed (fp:871-890); interims are idempotent for the
                # aggregator, so re-send the last one to re-arm the start strategies.
                await self._cancel_wait()
                # ``final_seen``: the turn now outlives its final, and re-arming with an
                # interim that final already superseded would open a ghost turn.
                if (
                    self._turn is not None
                    and not self._turn.final_seen
                    and self._turn.interim_frame is not None
                ):
                    self.stats["reemitted_interim"] += 1
                    f = self._turn.interim_frame
                    await self.push_frame(
                        InterimTranscriptionFrame(f.text, f.user_id, f.timestamp)
                    )
            await self.push_frame(frame, direction)
            return

        # DOWNSTREAM: STT-originated frames.
        if isinstance(frame, UserStartedSpeakingFrame):
            # Flux's StartOfTurn is the single reset point (§5.3-8/13): not the
            # UserStopped handler, since Flux's UserStopped overtakes its own final
            # (S10), and not the final either, since a second one may still extend it.
            if (
                self._turn is not None
                and self._turn.stop_seen
                and not self._turn.final_seen
            ):
                self.stats["turn_closed_without_final"] += 1
            await self._cancel_wait()
            self._turn = FluxTurn()
            await self._swallow_turn_signal(
                frame, direction, "swallowed_UserStartedSpeakingFrame"
            )
            return
        if isinstance(frame, UserStoppedSpeakingFrame):
            if self._turn is not None and not self._turn.final_seen:
                self._turn.stop_seen = True
            await self._swallow_turn_signal(
                frame, direction, "swallowed_UserStoppedSpeakingFrame"
            )
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
            await self._cancel_wait()
            held = await self._take_held()
            if held is not None and isinstance(frame, EndFrame):
                # Last chance to deliver speech the local VAD never claimed. On a
                # CancelFrame the pipeline is being torn down, so it is dropped instead.
                await self._release_held_as_message(held, FrameDirection.DOWNSTREAM)
            self._turn = None
            if self.stats:
                logger.info(f"{self}: hybrid turn stats {dict(self.stats)}")
        await self.push_frame(frame, direction)
