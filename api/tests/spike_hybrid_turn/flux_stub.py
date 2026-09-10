"""STT stub reproducing Deepgram Flux's frame-emission CALL PATTERN (spec §1.1), not a
pre-ordered frame list. Metadata frame identical to Flux: ttfs 0.0 + External strategies."""

import asyncio
from collections.abc import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601

from api.tests.spike_hybrid_turn.timeline import Timeline


class FluxStub(STTService):
    def __init__(self, timeline: Timeline, **kwargs):
        super().__init__(**kwargs)
        self._timeline = timeline
        self.emitted: list[tuple[float, str, str]] = []

    @property
    def supports_ttfs(self) -> bool:
        return False  # flux/stt.py:243-247

    def service_metadata_frame(self) -> STTMetadataFrame:
        frame = (
            super().service_metadata_frame()
        )  # ttfs_p99_latency == 0.0 (stt_service.py:554-561)
        frame.user_turn_strategies = (
            ExternalUserTurnStrategies()
        )  # flux/base.py:237-247
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

    async def emit_start_of_turn(self) -> None:
        # flux/base.py:690-691 — broadcast only; Dograh passes should_interrupt=False.
        self._log("StartOfTurn")
        await self.broadcast_frame(UserStartedSpeakingFrame)

    async def emit_eager(self, text: str) -> None:
        # flux/base.py:832
        self._log("EagerEndOfTurn", text)
        await self.push_frame(InterimTranscriptionFrame(text, "", time_now_iso8601()))

    async def emit_turn_resumed(self) -> None:
        self._log("TurnResumed")  # flux/base.py:698-709 — nothing reaches the pipeline

    async def emit_end_of_turn(
        self,
        text: str,
        *,
        yield_between_final_and_stop: bool = False,
        suppress_transcript: bool = False,
    ) -> None:
        # flux/base.py:766-794 — final (unless min_confidence drops it) then UserStopped.
        # Default (False): nothing between the two, the spec §1.1 idealisation. True mimics
        # the real service's `await self._handle_transcription(...)` (flux/base.py:792) and
        # `await self.stop_processing_metrics()` (:793) that sit before the broadcast (:794),
        # with one loop yield (red team F15 / register M5).
        self._log("EndOfTurn", text)
        if not suppress_transcript:
            await self.push_frame(
                TranscriptionFrame(text, "", time_now_iso8601(), finalized=True)
            )
        if yield_between_final_and_stop:
            await asyncio.sleep(0)
        await self.broadcast_frame(UserStoppedSpeakingFrame)
