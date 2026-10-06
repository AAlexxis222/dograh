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
from pipecat.frames.frames import TTSSpeakFrame
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

from api.services.pipecat.speech_playback import PlaybackOutcome
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.pipecat_engine_custom_tools import CustomToolManager
from api.services.workflow.pipecat_engine_variable_extractor import (
    VariableExtractionManager,
)
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.pipecat_test_utils import run_engine_test_pipeline
from pipecat.tests import MockLLMService, MockTTSService

_engines_under_test: list[PipecatEngine] = []


@pytest.fixture(autouse=True)
async def cleanup_engines():
    """Tear down every engine a test built.

    Pending speech arms a deadline timer, so without this it outlives the test
    that queued it.
    """
    yield
    for engine in _engines_under_test:
        await engine.cleanup()
    _engines_under_test.clear()


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
    engine.call_worker = task
    _engines_under_test.append(engine)

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
        llm=engine.active_agent.llm,
        pipeline_worker=engine.call_worker,
        context=engine.context,
        result_callback=AsyncMock(),
    )


async def _user_is_muted(engine: PipecatEngine) -> bool:
    """The engine's mute verdict for the next user frame."""
    return await engine.should_mute_user(TTSSpeakFrame("anything"))


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


def _http_tool_with_audio() -> SimpleNamespace:
    return SimpleNamespace(
        definition={
            "config": {
                "customMessageType": "audio",
                "customMessageRecordingId": "42",
            }
        }
    )


class TestQueuedSpeechMuteOwnership:
    """Queued speech mutes the user, and each operation releases only its own.

    VOZ-OLA0-F6 (fork PRs #10/#15) ported to ``SpeechPlaybackTracker``: the
    mute belongs to the speech request, so a request that never reaches the
    caller must not leave the user muted for the rest of the call, and
    releasing it must not unmute speech that a parallel function call (pipecat
    runs them concurrently) is still playing.
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

        assert engine.speech_playback.mutes_user is False
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

        assert engine.speech_playback.mutes_user is False
        params.result_callback.assert_awaited_once()

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
        handler = manager._create_http_tool_handler(
            _http_tool_with_audio(), "lookup_order"
        )
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

        assert engine.speech_playback.mutes_user is False
        assert await _user_is_muted(engine) is False
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
        handler = manager._create_http_tool_handler(
            _http_tool_with_audio(), "lookup_order"
        )
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

        assert engine.speech_playback.mutes_user is False
        assert await _user_is_muted(engine) is False
        params.result_callback.assert_awaited_once_with({"status": "ok"})

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
        other = engine.speech_playback.create(mute_user=True)

        await _run_transition_with_audio(engine)

        assert engine.speech_playback.mutes_user is True

        # The speech that is playing releases its own mute when it ends.
        other.finish(PlaybackOutcome.PLAYED)

        assert engine.speech_playback.mutes_user is False

    @pytest.mark.asyncio
    async def test_playback_queue_failure_before_audio_releases_mute(
        self, simple_workflow: WorkflowGraph
    ):
        """Failing before the audio frame has queued nothing audible.

        The recording's boundary markers and audio are queued in order; if the
        queue fails first, the end marker that would release the mute can never
        arrive, so the request releases its own mute.
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

        assert engine.speech_playback.mutes_user is False
        assert await _user_is_muted(engine) is False

    @pytest.mark.asyncio
    async def test_playback_queue_failure_after_audio_releases_mute(
        self, simple_workflow: WorkflowGraph
    ):
        """DIVERGENCE from the fork's F6 (VOZ-UNV-B9-02, see g0-cierre.md §2).

        The fork kept the user muted when a frame failed once the audio was
        queued, releasing on ``BotStoppedSpeakingFrame``. The tracker does not
        use bot activity events: a request whose queueing failed can never see
        its end marker, so it resolves FAILED and releases its mute at once
        (upstream ``test_enqueue_failure_releases_its_mute``). Never leaking the
        mute wins over muting the tail of an utterance on a failed transport.
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
                    side_effect=[None, None, None, RuntimeError("transport gone")]
                )
            )
        )

        await _run_transition_with_audio(engine)

        assert engine.speech_playback.mutes_user is False
        assert await _user_is_muted(engine) is False
