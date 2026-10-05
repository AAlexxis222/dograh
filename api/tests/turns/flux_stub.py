"""STT stub reproducing Deepgram Flux's frame-emission CALL PATTERN in Pipecat 1.12
(``pipecat/src/pipecat/services/deepgram/flux/stt_base.py``), not a pre-ordered frame list.
Metadata frame identical to Flux: ttfs 0.0 + External strategies."""

import asyncio
from collections.abc import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601

from api.tests.turns.timeline import Timeline


class FluxStub(STTService):
    def __init__(self, timeline: Timeline, **kwargs):
        super().__init__(**kwargs)
        self._timeline = timeline
        self.emitted: list[tuple[float, str, str]] = []

    @property
    def supports_ttfs(self) -> bool:
        return False

    def service_metadata_frame(self) -> STTMetadataFrame:
        frame = (
            super().service_metadata_frame()
        )  # ttfs_p99_latency == 0.0 (stt_service.py:554-561)
        frame.user_turn_strategies = ExternalUserTurnStrategies()  # stt_base.py:308-320
        return frame

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._timeline.t0 is None:
            self._timeline.start()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None

    def _log(self, kind: str, text: str = "") -> None:
        t = self._timeline.now_ms() if self._timeline.t0 is not None else -1.0
        self.emitted.append((t, kind, text))

    async def _push_partial(self, text: str) -> None:
        # stt_base.py:970-975 (``_push_partial_transcript``): blank text pushes nothing.
        if text:
            await self.push_frame(
                InterimTranscriptionFrame(text, "", time_now_iso8601())
            )

    async def emit_start_of_turn(self, text: str = "") -> None:
        # stt_base.py:801-820 — a proposal (broadcast) plus the first words, if any.
        self._log("StartOfTurn", text)
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        await self._push_partial(text)

    async def emit_update(self, text: str) -> None:
        # stt_base.py:948-968 — every Update is an interim, repeats included.
        self._log("Update", text)
        await self._push_partial(text)

    async def emit_turn_resumed(self) -> None:
        self._log("TurnResumed")  # stt_base.py:822-834 — nothing reaches the pipeline

    async def emit_end_of_turn(
        self,
        text: str,
        *,
        yield_between_final_and_stop: bool = False,
        suppress_transcript: bool = False,
    ) -> None:
        # stt_base.py:867-922 — final (unless min_confidence drops it), then the stop
        # proposal: a ControlFrame, so it stays BEHIND the final (frames.py:1393-1405).
        # True mimics the ``await self._handle_transcription(...)`` that sits between the two
        # (:920), with one loop yield.
        self._log("EndOfTurn", text)
        if not suppress_transcript:
            await self.push_frame(
                TranscriptionFrame(text, "", time_now_iso8601(), finalized=True)
            )
        if yield_between_final_and_stop:
            await asyncio.sleep(0)
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
