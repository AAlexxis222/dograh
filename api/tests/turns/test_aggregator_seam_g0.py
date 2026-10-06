"""``build_context_aggregators`` is the only place a production call builds its
aggregator pair, and the hybrid user aggregator keeps its behaviour when the
``GreetingController`` swaps the start strategies around the greeting (VOZ-G0-07)."""

import ast
import asyncio
from pathlib import Path

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
)
from pipecat.tests import run_test
from pipecat.tests.utils import SleepFrame
from pipecat.turns.user_start import (
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from api.services.pipecat import run_pipeline
from api.services.pipecat.speech_playback import PlaybackOutcome
from api.services.pipecat.turns.hybrid_user_aggregator import HybridUserAggregator

_SETTLE = 0.05


def _strategies():
    return UserTurnStrategies(
        start=[TranscriptionUserTurnStartStrategy()],
        stop=[ExternalUserTurnStopStrategy(wait_for_transcript=False)],
    )


def _build(llm_context, hybrid, strategies=None):
    return run_pipeline.build_context_aggregators(
        llm_context,
        user_params=run_pipeline.build_user_aggregator_params(
            user_turn_strategies=strategies or _strategies(), user_turn_stop_timeout=5.0
        ),
        assistant_params=LLMAssistantAggregatorParams(),
        realtime_service_mode=False,
        hybrid=hybrid,
    )


def test_run_pipeline_exposes_the_hybrid_seam(llm_context):
    # `run_pipeline.build_context_aggregators` is the name the pipeline builds through.
    assert isinstance(_build(llm_context, True).user(), HybridUserAggregator)


def test_no_production_code_builds_a_pair_outside_the_seam():
    # A construction site that bypasses the factory silently runs upstream's user
    # aggregator on a hybrid workflow (realtime and text chat included).
    api_root = Path(run_pipeline.__file__).parents[2]
    seam = api_root / "services" / "pipecat" / "turns" / "hybrid_user_aggregator.py"
    banned = {"LLMContextAggregatorPair", "LLMUserAggregator", "HybridUserAggregator"}
    offenders = []
    for path in (api_root / "services").rglob("*.py"):
        if path == seam:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", None))
                if name in banned:
                    offenders.append(f"{path.relative_to(api_root)}:{node.lineno}")
    assert offenders == []


def test_non_hybrid_source_keeps_upstream_pair(llm_context):
    pair = _build(llm_context, False)
    assert isinstance(pair, LLMContextAggregatorPair)
    assert not isinstance(pair.user(), HybridUserAggregator)


@pytest.mark.asyncio
async def test_greeting_swap_keeps_hybrid_behaviour_then_restores_strategies(
    llm_context, greeting_controller, playback
):
    strategies = _strategies()
    original_start = list(strategies.start)
    original_stop = list(strategies.stop)
    user = _build(llm_context, True, strategies).user()
    greeting_controller.bind(user)
    greeting = playback.create(greeting=True)
    seen_during_greeting = {}

    async def _probe_then_finish():
        await asyncio.sleep(0.4)
        current = user.user_turn_controller.user_turn_strategies
        seen_during_greeting["start"] = list(current.start)
        seen_during_greeting["stop"] = list(current.stop)
        greeting.finish(PlaybackOutcome.PLAYED)

    probe = asyncio.create_task(_probe_then_finish())
    # While the greeting is pending the start strategy is MinWords(2) and resets the
    # aggregation on every shorter transcript; the one-word delta of an open turn must
    # still land in the pending text (the hybrid override), not wipe it.
    await run_test(
        user,
        frames_to_send=[
            UserStartedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
            TranscriptionFrame("quiero reservar para el", "u1", "t", finalized=False),
            SleepFrame(sleep=_SETTLE),
            BotStartedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
            TranscriptionFrame("sábado.", "u1", "t", finalized=True),
            SleepFrame(sleep=_SETTLE),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=_SETTLE),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.6),
        ],
        expected_down_frames=None,
    )
    await probe
    assert isinstance(user, HybridUserAggregator)
    assert isinstance(seen_during_greeting["start"][0], MinWordsUserTurnStartStrategy)
    assert seen_during_greeting["stop"] == original_stop
    restored = user.user_turn_controller.user_turn_strategies
    assert list(restored.start) == original_start
    assert list(restored.stop) == original_stop
    assert [
        m["content"] for m in llm_context.get_messages() if m["role"] == "user"
    ] == ["quiero reservar para el sábado."]
