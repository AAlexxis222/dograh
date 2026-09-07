"""Tests verifying user is muted while a transition function is executing.

When the LLM calls a transition function (registered via
``_register_transition_function_with_llm``), pipecat broadcasts a
``FunctionCallsStartedFrame`` that ``FunctionCallUserMuteStrategy`` uses to
mute the user until a ``FunctionCallResultFrame`` arrives. These tests assert
that mute behavior holds end-to-end through the engine's transition flow,
so that user audio doesn't race the node switch / extraction / context update
that runs inside the transition function.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.tests.mock_transport import MockTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_mute import (
    CallbackUserMuteStrategy,
    FunctionCallUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)

from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.pipecat_engine_custom_tools import CustomToolManager
from api.services.workflow.pipecat_engine_variable_extractor import (
    VariableExtractionManager,
)
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.pipecat_test_utils import run_engine_test_pipeline
from pipecat.tests import MockLLMService, MockTTSService


async def _build_engine_and_pipeline(
    workflow: WorkflowGraph,
    mock_llm: MockLLMService,
):
    """Set up engine + pipeline mirroring the non-realtime production wiring.

    Returns (engine, transport, task, function_call_mute_strategy,
    user_context_aggregator).
    """
    tts = MockTTSService(mock_audio_duration_ms=40, frame_delay=0)

    transport = MockTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_end_silence_secs=0,
        ),
    )

    context = LLMContext()

    engine = PipecatEngine(
        llm=mock_llm,
        context=context,
        workflow=workflow,
        call_context_vars={"customer_name": "Test User"},
        workflow_run_id=1,
    )

    # Hold a reference so the test can introspect the in-progress set.
    function_call_mute_strategy = FunctionCallUserMuteStrategy()

    # Match run_pipeline.py's non-realtime mute-strategy stack so the test
    # exercises the same wiring that would be active in a real call.
    user_mute_strategies = [
        MuteUntilFirstBotCompleteUserMuteStrategy(),
        function_call_mute_strategy,
        CallbackUserMuteStrategy(should_mute_callback=engine.should_mute_user),
    ]

    user_params = LLMUserAggregatorParams(user_mute_strategies=user_mute_strategies)
    assistant_params = LLMAssistantAggregatorParams()

    context_aggregator = LLMContextAggregatorPair(
        context, assistant_params=assistant_params, user_params=user_params
    )
    user_context_aggregator = context_aggregator.user()
    assistant_context_aggregator = context_aggregator.assistant()

    pipeline = Pipeline(
        [
            transport.input(),
            user_context_aggregator,
            mock_llm,
            tts,
            transport.output(),
            assistant_context_aggregator,
        ]
    )

    task = PipelineWorker(pipeline, params=PipelineParams(), enable_rtvi=False)
    engine.set_task(task)

    return (
        engine,
        transport,
        task,
        function_call_mute_strategy,
        user_context_aggregator,
    )


class TestTransitionFunctionMutesUser:
    """Verify the user is muted while transition functions execute."""

    @pytest.mark.asyncio
    async def test_user_is_muted_during_transition_function(
        self, simple_workflow: WorkflowGraph
    ):
        """The user must be muted from the moment a transition function starts
        until its result is delivered.

        Scenario:
        1. LLM calls the ``end_call`` transition function (start → end edge).
        2. Wrap the registered handler so we can read mute state from inside it.
        3. VERIFY: the function-call mute strategy has the call in flight.
        4. VERIFY: the user aggregator's ``_user_is_muted`` flag is True.
        """
        step_0_chunks = MockLLMService.create_function_call_chunks(
            function_name="end_call",
            arguments={},
            tool_call_id="call_end_1",
        )
        llm = MockLLMService(mock_steps=[step_0_chunks], chunk_delay=0.001)

        (
            engine,
            transport,
            task,
            function_call_mute_strategy,
            user_context_aggregator,
        ) = await _build_engine_and_pipeline(simple_workflow, llm)

        captured_states: list[dict] = []

        # Wrap register_function so we can introspect mute state from inside
        # the transition handler. We must wrap *after* the engine is created
        # but *before* set_node registers the transition functions.
        original_register_function = llm.register_function

        def wrapping_register_function(name, func, *args, **kwargs):
            async def wrapped(function_call_params):
                # Yield once so the user aggregator has a chance to drain
                # the broadcasted FunctionCallsStartedFrame and update its
                # mute state before we sample it.
                await asyncio.sleep(0.02)
                captured_states.append(
                    {
                        "name": name,
                        "function_call_in_progress": bool(
                            function_call_mute_strategy._function_call_in_progress
                        ),
                        "user_is_muted": user_context_aggregator._user_is_muted,
                        "tool_call_ids": set(
                            function_call_mute_strategy._function_call_in_progress
                        ),
                    }
                )
                return await func(function_call_params)

            return original_register_function(name, wrapped, *args, **kwargs)

        llm.register_function = wrapping_register_function

        with patch(
            "api.db:db_client.get_organization_id_by_workflow_run_id",
            new_callable=AsyncMock,
            return_value=1,
        ):
            with patch.object(
                VariableExtractionManager,
                "_perform_extraction",
                new_callable=AsyncMock,
                return_value={"user_intent": "end call"},
            ):
                await run_engine_test_pipeline(task, engine, transport)

        assert len(captured_states) == 1, (
            f"Expected the transition function to be invoked exactly once, "
            f"got {len(captured_states)}: {captured_states}"
        )
        state = captured_states[0]
        assert state["name"] == "end_call"
        assert state["function_call_in_progress"], (
            "FunctionCallUserMuteStrategy should have the transition call in "
            f"progress while the handler runs (state={state})"
        )
        assert "call_end_1" in state["tool_call_ids"], (
            f"Expected tool_call_id 'call_end_1' to be tracked, got {state['tool_call_ids']}"
        )
        assert state["user_is_muted"], (
            "User aggregator's _user_is_muted should be True during the "
            f"transition function (state={state})"
        )

    @pytest.mark.asyncio
    async def test_user_is_unmuted_after_transition_function_returns(
        self, simple_workflow: WorkflowGraph
    ):
        """After the transition function's result is delivered, the function-call
        mute strategy should clear its in-progress set. Other strategies in the
        stack (CallbackUserMuteStrategy via engine.should_mute_user) may still
        keep the pipeline muted because end_call_with_reason fires when the
        engine reaches the End node, but the function-call strategy itself
        must release its hold.
        """
        step_0_chunks = MockLLMService.create_function_call_chunks(
            function_name="end_call",
            arguments={},
            tool_call_id="call_end_1",
        )
        llm = MockLLMService(mock_steps=[step_0_chunks], chunk_delay=0.001)

        (
            engine,
            transport,
            task,
            function_call_mute_strategy,
            _user_context_aggregator,
        ) = await _build_engine_and_pipeline(simple_workflow, llm)

        with patch(
            "api.db:db_client.get_organization_id_by_workflow_run_id",
            new_callable=AsyncMock,
            return_value=1,
        ):
            with patch.object(
                VariableExtractionManager,
                "_perform_extraction",
                new_callable=AsyncMock,
                return_value={"user_intent": "end call"},
            ):
                await run_engine_test_pipeline(task, engine, transport)

        assert function_call_mute_strategy._function_call_in_progress == set(), (
            "FunctionCallUserMuteStrategy should have cleared its in-progress "
            "set after the transition function's result was delivered, got "
            f"{function_call_mute_strategy._function_call_in_progress}"
        )


def _function_call_params(engine: PipecatEngine, name: str) -> FunctionCallParams:
    return FunctionCallParams(
        function_name=name,
        tool_call_id=f"call_{name}",
        arguments={},
        llm=engine.llm,
        pipeline_worker=engine.task,
        context=engine.context,
        result_callback=AsyncMock(),
    )


def _mute_holds(engine: PipecatEngine) -> set[int]:
    """Every queued-speech hold currently keeping the user muted."""
    return engine._queued_speech_mute_pending | engine._queued_speech_mute_playing


async def _run_transition_with_audio(engine: PipecatEngine) -> FunctionCallParams:
    """Run a transition that plays recording 42 as its transition speech."""
    transition = await engine._create_transition_func(
        "end_call",
        "end",
        transition_speech_type="audio",
        transition_speech_recording_id="42",
    )
    params = _function_call_params(engine, "end_call")

    with (
        patch.object(
            engine, "_perform_variable_extraction_if_needed", new_callable=AsyncMock
        ),
        patch.object(engine, "set_node", new_callable=AsyncMock),
    ):
        await transition(params)

    return params


class TestQueuedSpeechMuteOwnership:
    """Queued speech mutes the user, and each operation releases only its own.

    A hold is taken before the recording is fetched. If the fetch yields no
    audio nothing is queued, so the ``BotStoppedSpeakingFrame`` that normally
    ends the hold never arrives and the user would stay muted for the rest of
    the call. Releasing it must not unmute speech that a parallel function
    call (pipecat runs them concurrently) is still playing.
    """

    @pytest.mark.asyncio
    async def test_transition_audio_fetch_failure_releases_mute(
        self, simple_workflow: WorkflowGraph
    ):
        llm = MockLLMService(mock_steps=[], chunk_delay=0.001)
        engine, _transport, _task, _strategy, _agg = await _build_engine_and_pipeline(
            simple_workflow, llm
        )
        engine.set_fetch_recording_audio(AsyncMock(return_value=None))

        params = await _run_transition_with_audio(engine)

        assert _mute_holds(engine) == set()
        params.result_callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_transition_audio_fetch_exception_releases_mute(
        self, simple_workflow: WorkflowGraph
    ):
        llm = MockLLMService(mock_steps=[], chunk_delay=0.001)
        engine, _transport, _task, _strategy, _agg = await _build_engine_and_pipeline(
            simple_workflow, llm
        )
        engine.set_fetch_recording_audio(
            AsyncMock(side_effect=RuntimeError("recording service down"))
        )

        params = await _run_transition_with_audio(engine)

        assert _mute_holds(engine) == set()
        params.result_callback.assert_awaited_once_with(
            {"status": "error", "error": "recording service down"}
        )

    @pytest.mark.asyncio
    async def test_http_tool_audio_fetch_failure_releases_mute(
        self, simple_workflow: WorkflowGraph
    ):
        llm = MockLLMService(mock_steps=[], chunk_delay=0.001)
        engine, _transport, _task, _strategy, _agg = await _build_engine_and_pipeline(
            simple_workflow, llm
        )
        engine.set_fetch_recording_audio(AsyncMock(return_value=None))

        manager = CustomToolManager(engine)
        tool = SimpleNamespace(
            definition={
                "config": {
                    "customMessageType": "audio",
                    "customMessageRecordingId": "42",
                }
            }
        )
        handler = manager._create_http_tool_handler(tool, "lookup_order")
        params = _function_call_params(engine, "lookup_order")

        with (
            patch(
                "api.services.workflow.pipecat_engine_custom_tools.execute_http_tool",
                new_callable=AsyncMock,
                return_value={"status": "ok"},
            ),
            patch.object(
                manager, "get_organization_id", new_callable=AsyncMock, return_value=1
            ),
        ):
            await handler(params)

        assert _mute_holds(engine) == set()
        params.result_callback.assert_awaited_once_with({"status": "ok"})

    @pytest.mark.asyncio
    async def test_http_tool_audio_fetch_exception_releases_mute(
        self, simple_workflow: WorkflowGraph
    ):
        llm = MockLLMService(mock_steps=[], chunk_delay=0.001)
        engine, _transport, _task, _strategy, _agg = await _build_engine_and_pipeline(
            simple_workflow, llm
        )
        engine.set_fetch_recording_audio(
            AsyncMock(side_effect=RuntimeError("recording service down"))
        )

        manager = CustomToolManager(engine)
        tool = SimpleNamespace(
            definition={
                "config": {
                    "customMessageType": "audio",
                    "customMessageRecordingId": "42",
                }
            }
        )
        handler = manager._create_http_tool_handler(tool, "lookup_order")
        params = _function_call_params(engine, "lookup_order")

        with patch.object(
            manager, "get_organization_id", new_callable=AsyncMock, return_value=1
        ):
            await handler(params)

        assert _mute_holds(engine) == set()
        params.result_callback.assert_awaited_once_with(
            {"status": "error", "error": "recording service down"}
        )

    @pytest.mark.asyncio
    async def test_fetch_failure_keeps_parallel_playback_muted(
        self, simple_workflow: WorkflowGraph
    ):
        """A failed operation must not unmute another one's speech.

        Function calls run in parallel, so a transition whose recording never
        arrives can finish while a second one is still being played out.
        """
        llm = MockLLMService(mock_steps=[], chunk_delay=0.001)
        engine, _transport, _task, _strategy, _agg = await _build_engine_and_pipeline(
            simple_workflow, llm
        )
        engine.set_fetch_recording_audio(AsyncMock(return_value=None))

        # Another function call already queued its speech.
        engine.mute_until_speech_playback_ends()

        await _run_transition_with_audio(engine)

        assert await engine.should_mute_user(TTSSpeakFrame("anything")) is True

        # The speech that is playing releases its own hold when it ends.
        await engine.should_mute_user(BotStoppedSpeakingFrame())

        assert _mute_holds(engine) == set()
        assert await engine.should_mute_user(TTSSpeakFrame("anything")) is False

    @pytest.mark.asyncio
    async def test_partial_playback_queue_keeps_user_muted(
        self, simple_workflow: WorkflowGraph
    ):
        """A frame failing mid-utterance must not unmute the queued audio.

        ``play_audio`` pushes several frames. Once the first has reached the
        transport the utterance is on its way to the caller, so the hold
        belongs to playback even though the call raised.
        """
        llm = MockLLMService(mock_steps=[], chunk_delay=0.001)
        engine, _transport, _task, _strategy, _agg = await _build_engine_and_pipeline(
            simple_workflow, llm
        )
        engine.set_fetch_recording_audio(
            AsyncMock(
                return_value=SimpleNamespace(audio=b"\x00\x00", transcript="hello")
            )
        )
        engine.set_transport_output(
            SimpleNamespace(
                queue_frame=AsyncMock(
                    side_effect=[None, RuntimeError("transport gone")]
                )
            )
        )

        await _run_transition_with_audio(engine)

        assert engine._queued_speech_mute_pending == set()
        assert len(engine._queued_speech_mute_playing) == 1
        assert await engine.should_mute_user(TTSSpeakFrame("anything")) is True

    @pytest.mark.asyncio
    async def test_wait_for_speech_playback_start_timeout_keeps_other_holds(
        self, simple_workflow: WorkflowGraph
    ):
        """The transfer path arms playback without taking a hold of its own."""
        llm = MockLLMService(mock_steps=[], chunk_delay=0.001)
        engine, _transport, _task, _strategy, _agg = await _build_engine_and_pipeline(
            simple_workflow, llm
        )
        engine.mute_until_speech_playback_ends()
        engine.arm_speech_playback()

        played = await engine.wait_for_speech_playback(start_timeout=0.01)

        assert played is False
        assert await engine.should_mute_user(TTSSpeakFrame("anything")) is True

    @pytest.mark.asyncio
    async def test_wait_for_speech_playback_finish_timeout_keeps_other_holds(
        self, simple_workflow: WorkflowGraph
    ):
        """Playback started but never finished: the hold still belongs to it."""
        llm = MockLLMService(mock_steps=[], chunk_delay=0.001)
        engine, _transport, _task, _strategy, _agg = await _build_engine_and_pipeline(
            simple_workflow, llm
        )
        engine.mute_until_speech_playback_ends()
        engine.arm_speech_playback()
        await engine.should_mute_user(BotStartedSpeakingFrame())

        played = await engine.wait_for_speech_playback(
            start_timeout=1.0, playback_timeout=0.01
        )

        assert played is False
        assert await engine.should_mute_user(TTSSpeakFrame("anything")) is True
