"""Point B (spec B2 VOZ-AC-B2-03): ``stt → answer_supervisor → absorber → user_aggregator →
answer_supervisor.llm_gate() → call_monitor → generation_stage``, and the two passthrough
preconditions it rests on (VOZ-PB-06 (a)(b))."""

import pytest
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    InterruptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor

from api.schemas.answer_supervisor import AnswerSupervisorConfig
from api.services.pipecat.agent_bridge import AgentBridgeProcessor
from api.services.pipecat.pipeline_builder import build_pipeline
from api.services.pipecat.processors.answer_supervisor import AnswerSupervisor
from api.services.pipecat.turns.absorber import TurnSignalAbsorberProcessor
from api.services.pipecat.turns.hybrid_user_aggregator import build_context_aggregators
from api.tests.turns.fakes import user_frames_out_of


class _Transport:
    def input(self):
        return FrameProcessor(name="in")

    def output(self):
        return FrameProcessor(name="out")


def _supervisor() -> AnswerSupervisor:
    return AnswerSupervisor(AnswerSupervisorConfig(), context=LLMContext())


def test_absorber_between_answer_supervisor_and_user_aggregator():
    pair = build_context_aggregators(
        LLMContext(),
        user_params=LLMUserAggregatorParams(),
        assistant_params=LLMAssistantAggregatorParams(),
        realtime_service_mode=None,
        hybrid=True,
    )
    pipeline = build_pipeline(
        _Transport(),
        FrameProcessor(name="stt"),
        FrameProcessor(name="audio"),
        pair.user(),
        pair.assistant(),
        FrameProcessor(name="monitor"),
        [FrameProcessor(name="generation")],
        FrameProcessor(name="metrics"),
        FrameProcessor(name="funnel"),
        answer_supervisor=_supervisor(),
        turn_signal_absorber=TurnSignalAbsorberProcessor(),
    )
    names = [type(p).__name__ for p in pipeline._processors[1:-1]]
    assert names[2:7] == [
        "FrameProcessor",  # stt
        "AnswerSupervisor",
        "TurnSignalAbsorberProcessor",
        "HybridUserAggregator",
        "AnswerContextGate",
    ]


# What reaches point B from the STT, plus a downstream copy of every broadcast a turn engine
# reads. Fresh instances per run: identity is what "untouched" is checked against.
def _user_frames():
    return [
        ProposedUserStartedSpeakingFrame(),
        VADUserStartedSpeakingFrame(),
        InterimTranscriptionFrame("hola", "u", "t"),
        TranscriptionFrame("hola quiero", "u", "t", finalized=True),
        ProposedUserStoppedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
    ]


@pytest.mark.asyncio
async def test_answer_supervisor_passes_user_frames_untouched():  # VOZ-PB-06 (a)
    before, after = _user_frames(), _user_frames()
    # ``release``: the state the supervisor stays in once it has classified the answer.
    out = await user_frames_out_of(
        lambda _runner: _supervisor(), [*before, lambda s: s.release(), *after]
    )
    assert [id(f) for f in out] == [id(f) for f in before + after]


@pytest.mark.asyncio
async def test_generation_stage_does_not_modify_user_frames():  # VOZ-PB-06 (b)
    # Downstream of the aggregator only its broadcasts travel on (the aggregator consumes
    # every transcript, ``llm_response_universal.py:847-861``). The bridge tees them to the
    # call side as they are (``agent_bridge.py:35-42, :82-85``).
    frames = [
        UserStartedSpeakingFrame(),
        InterruptionFrame(),
        UserStoppedSpeakingFrame(),
    ]
    out = await user_frames_out_of(
        lambda runner: AgentBridgeProcessor(
            bus=runner.bus,
            worker_name="call",
            selected_visit=lambda: "agent",
            allow_inference=lambda: True,
        ),
        frames,
    )
    assert [id(f) for f in out] == [id(f) for f in frames]
