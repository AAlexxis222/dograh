"""User aggregator for the hybrid turn mode (spec §6.1.1).

Identical to ``LLMUserAggregator`` except that a ``TranscriptionReplaceFrame`` replaces the
pending aggregation. ``build_context_aggregators`` is the single seam ``run_pipeline`` uses
so the stock ``LLMContextAggregatorPair`` stays untouched when the hybrid is off.
"""

from api.services.pipecat.turns.frames import TranscriptionReplaceFrame
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection


class HybridUserAggregator(LLMUserAggregator):
    """The stock user aggregator, plus the replace frame the absorber may send it."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TranscriptionReplaceFrame):
            # Drop what the promoted interim put in the pending aggregation, then let the
            # rewritten final take the normal transcription path.
            replacement = TranscriptionFrame(
                frame.text, frame.user_id, frame.timestamp, finalized=True
            )
            # Only when that path will actually aggregate the replacement: the base drops
            # blank transcriptions (`_handle_transcription`) and every transcription while
            # the user is muted (`_maybe_mute_frame`), and a replace must never destroy
            # more than it replaces.
            if replacement.text.strip() and not self._user_is_muted:
                await self.reset()
            frame = replacement
        await super().process_frame(frame, direction)


class HybridContextAggregators:
    """The two halves the pipeline needs, built like ``LLMContextAggregatorPair`` does
    (``llm_response_universal.py`` pair constructor) but with the hybrid user half."""

    def __init__(
        self,
        context: LLMContext,
        *,
        user_params: LLMUserAggregatorParams,
        assistant_params: LLMAssistantAggregatorParams,
        realtime_service_mode: bool | None,
    ):
        self._user = HybridUserAggregator(
            context, params=user_params, _realtime_service_mode=realtime_service_mode
        )
        self._assistant = LLMAssistantAggregator(
            context,
            params=assistant_params,
            _realtime_service_mode=realtime_service_mode,
            _paired_user_aggregator=self._user,
        )

    def user(self) -> HybridUserAggregator:
        return self._user

    def assistant(self) -> LLMAssistantAggregator:
        return self._assistant


def build_context_aggregators(
    context: LLMContext,
    *,
    user_params: LLMUserAggregatorParams,
    assistant_params: LLMAssistantAggregatorParams,
    realtime_service_mode: bool | None,
    hybrid: bool,
) -> LLMContextAggregatorPair | HybridContextAggregators:
    if not hybrid:
        return LLMContextAggregatorPair(
            context,
            assistant_params=assistant_params,
            user_params=user_params,
            realtime_service_mode=realtime_service_mode,
        )
    return HybridContextAggregators(
        context,
        user_params=user_params,
        assistant_params=assistant_params,
        realtime_service_mode=realtime_service_mode,
    )
