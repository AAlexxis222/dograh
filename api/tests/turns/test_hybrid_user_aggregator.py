"""A rewrite from the STT replaces the pending aggregation instead of being appended to it
(D-11, README 8a §5.3-14). Everything else is the stock user aggregator."""

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    UninterruptibleFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.turns.user_start import (
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from api.services.pipecat.turns.frames import TranscriptionReplaceFrame
from api.services.pipecat.turns.hybrid_user_aggregator import (
    HybridUserAggregator,
    build_context_aggregators,
)
from pipecat.tests import run_test

# Each frame is let settle before the next is queued: the turn start broadcasts an
# interruption, and an interruption cancels the data frames still queued behind it.
_SETTLE = 0.05


def _params(mute_strategies=None, start_strategies=None):
    # The external strategy closes the turn on the ``UserStoppedSpeakingFrame`` below, which
    # is what a hybrid pipeline does (local turn detection signals the stop). The
    # speech-timeout strategy was measured first: with no VAD the turn starts and never
    # stops inside ``run_test``, so the aggregation would only reach the context through the
    # EndFrame teardown.
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=start_strategies or [TranscriptionUserTurnStartStrategy()],
            stop=[ExternalUserTurnStopStrategy(wait_for_transcript=False)],
        ),
        user_mute_strategies=mute_strategies or [],
        user_turn_stop_timeout=5.0,
    )


def _build(hybrid, mute_strategies=None, start_strategies=None):
    context = LLMContext(messages=[{"role": "system", "content": "t"}])
    aggs = build_context_aggregators(
        context,
        user_params=_params(mute_strategies, start_strategies),
        assistant_params=LLMAssistantAggregatorParams(),
        realtime_service_mode=False,
        hybrid=hybrid,
    )
    return context, aggs


def test_non_hybrid_returns_stock_pair():
    _, aggs = _build(False)
    assert isinstance(aggs, LLMContextAggregatorPair)
    assert not isinstance(aggs.user(), HybridUserAggregator)


def test_replace_frame_survives_an_interruption():
    # An interruption empties the processing queue of everything that is not
    # uninterruptible (``FrameQueue.reset``), and the interim this frame corrects is
    # already in the aggregation.
    assert issubclass(TranscriptionReplaceFrame, UninterruptibleFrame)


def test_hybrid_returns_hybrid_user_and_paired_assistant():
    _, aggs = _build(True)
    assert isinstance(aggs.user(), HybridUserAggregator)
    assert aggs.assistant()._paired_user_aggregator is aggs.user()


@pytest.mark.asyncio
async def test_replace_frame_substitutes_pending_aggregation():
    context, aggs = _build(True)
    user = aggs.user()
    await run_test(
        user,
        frames_to_send=[
            UserStartedSpeakingFrame(),
            TranscriptionFrame("quiero reservar para el", "u1", "t", finalized=False),
            SleepFrame(sleep=_SETTLE),
            TranscriptionReplaceFrame("Quiero cancelar para el sábado.", "u1", "t"),
            SleepFrame(sleep=_SETTLE),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
        ],
        expected_down_frames=None,
    )
    user_messages = [m for m in context.get_messages() if m["role"] == "user"]
    assert [m["content"] for m in user_messages] == ["Quiero cancelar para el sábado."]


@pytest.mark.asyncio
async def test_blank_replace_keeps_the_pending_aggregation():
    # The base drops blank transcriptions, so resetting would delete the promoted interim
    # and put nothing in its place.
    context, aggs = _build(True)
    await run_test(
        aggs.user(),
        frames_to_send=[
            UserStartedSpeakingFrame(),
            TranscriptionFrame("quiero reservar para el", "u1", "t", finalized=False),
            SleepFrame(sleep=_SETTLE),
            TranscriptionReplaceFrame("   ", "u1", "t"),
            SleepFrame(sleep=_SETTLE),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
        ],
        expected_down_frames=None,
    )
    user_messages = [m for m in context.get_messages() if m["role"] == "user"]
    assert [m["content"] for m in user_messages] == ["quiero reservar para el"]


@pytest.mark.asyncio
async def test_replace_while_muted_keeps_the_pending_aggregation():
    # Same reasoning for the other guard: the base drops every transcription while the
    # user is muted. Mute is driven the way production drives it — bot speech.
    context, aggs = _build(True, mute_strategies=[AlwaysUserMuteStrategy()])
    await run_test(
        aggs.user(),
        frames_to_send=[
            UserStartedSpeakingFrame(),
            TranscriptionFrame("quiero reservar para el", "u1", "t", finalized=False),
            SleepFrame(sleep=_SETTLE),
            BotStartedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
            TranscriptionReplaceFrame("Quiero cancelar para el sábado.", "u1", "t"),
            SleepFrame(sleep=_SETTLE),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
        ],
        expected_down_frames=None,
    )
    user_messages = [m for m in context.get_messages() if m["role"] == "user"]
    assert [m["content"] for m in user_messages] == ["quiero reservar para el"]


def _min_words_frames(tail):
    # A turn is open with the promoted interim pending when the bot starts speaking (the
    # strategy clears its bot flag on a turn start, so the order matters); ``tail`` is the
    # one-word delta the absorber sends next, shorter than min_words=3.
    return [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=_SETTLE),
        TranscriptionFrame("quiero reservar para el", "u1", "t", finalized=False),
        SleepFrame(sleep=_SETTLE),
        BotStartedSpeakingFrame(),
        SleepFrame(sleep=_SETTLE),
        TranscriptionFrame(tail, "u1", "t", finalized=True),
        SleepFrame(sleep=_SETTLE),
        BotStoppedSpeakingFrame(),
        SleepFrame(sleep=_SETTLE),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=_SETTLE),
    ]


@pytest.mark.asyncio
async def test_short_delta_inside_an_open_turn_does_not_reset_the_aggregation():
    # min_words resets the aggregation on every transcript shorter than min_words while
    # the bot speaks; a token delta of an open turn is exactly that, and the reset would
    # wipe the promoted text before it.
    context, aggs = _build(
        True, start_strategies=[MinWordsUserTurnStartStrategy(min_words=3)]
    )
    await run_test(
        aggs.user(),
        frames_to_send=_min_words_frames("sábado."),
        expected_down_frames=None,
    )
    user_messages = [
        m["content"] for m in context.get_messages() if m["role"] == "user"
    ]
    assert user_messages == ["quiero reservar para el sábado."]


@pytest.mark.asyncio
async def test_short_transcript_with_no_turn_open_is_still_reset():
    # Control: with no turn open the base's rule stands, so noise while the bot speaks is
    # still discarded and never opens a turn.
    context, aggs = _build(
        True, start_strategies=[MinWordsUserTurnStartStrategy(min_words=3)]
    )
    await run_test(
        aggs.user(),
        frames_to_send=[
            BotStartedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
            TranscriptionFrame("eh", "u1", "t", finalized=True),
            SleepFrame(sleep=_SETTLE),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
        ],
        expected_down_frames=None,
    )
    assert [m for m in context.get_messages() if m["role"] == "user"] == []


@pytest.mark.asyncio
async def test_stock_aggregator_wipes_the_short_delta_which_is_the_bug_being_fixed():
    # Control: the stock aggregator loses the promoted text to the one-word delta.
    context, aggs = _build(
        False, start_strategies=[MinWordsUserTurnStartStrategy(min_words=3)]
    )
    await run_test(
        aggs.user(),
        frames_to_send=_min_words_frames("sábado."),
        expected_down_frames=None,
    )
    user_messages = [
        m["content"] for m in context.get_messages() if m["role"] == "user"
    ]
    assert "quiero reservar" not in " ".join(user_messages)


@pytest.mark.asyncio
async def test_stock_aggregator_concatenates_which_is_the_bug_being_fixed():
    # Control: the same frames through the stock pair produce the duplicated context.
    context, aggs = _build(False)
    user = aggs.user()
    await run_test(
        user,
        frames_to_send=[
            UserStartedSpeakingFrame(),
            TranscriptionFrame("quiero reservar para el", "u1", "t", finalized=False),
            SleepFrame(sleep=_SETTLE),
            TranscriptionFrame(
                "Quiero cancelar para el sábado.", "u1", "t", finalized=True
            ),
            SleepFrame(sleep=_SETTLE),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
        ],
        expected_down_frames=None,
    )
    user_messages = [m for m in context.get_messages() if m["role"] == "user"]
    assert len(user_messages) == 1
    assert "quiero reservar para el" in user_messages[0]["content"]
    assert "Quiero cancelar" in user_messages[0]["content"]
