"""turn.source=local with a server-turn STT wires the hybrid: local strategies, the absorber
at point B (right above the user aggregator, below the answer supervisor), the hybrid
aggregator. Everything else is exactly today's pipeline (spec §1.3 'expose ≠ change')."""

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_start import (
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)

from api.services.pipecat.pipeline_builder import build_pipeline
from api.services.pipecat.run_pipeline import (
    HybridTurn,
    _create_non_realtime_user_turn_start_strategies,
    _create_non_realtime_user_turn_stop_strategies,
    resolve_hybrid_turn,
)
from api.services.pipecat.turns.absorber import TurnSignalAbsorberProcessor


def test_hybrid_off_by_default():
    assert resolve_hybrid_turn({}, uses_external_turns=True, is_realtime=False) is None
    assert (
        resolve_hybrid_turn(
            {"turn": {"source": "auto"}}, uses_external_turns=True, is_realtime=False
        )
        is None
    )


def test_hybrid_needs_local_source_and_server_turn_stt():
    cfg = {"turn": {"source": "local", "hybrid": {"wait_ms": 0, "hold_ms": 800}}}
    assert resolve_hybrid_turn(
        cfg, uses_external_turns=True, is_realtime=False
    ) == HybridTurn(0, 800)
    # Nova: local is plain config
    assert (
        resolve_hybrid_turn(cfg, uses_external_turns=False, is_realtime=False) is None
    )
    assert resolve_hybrid_turn(cfg, uses_external_turns=True, is_realtime=True) is None


def test_hybrid_defaults_when_knobs_absent():
    assert resolve_hybrid_turn(
        {"turn": {"source": "local"}}, uses_external_turns=True, is_realtime=False
    ) == HybridTurn(0, 1500)


def test_stored_document_values_are_bounded_or_defaulted():
    # A document written straight to the store skips the schema, and the cascade clamps
    # numbers only: a string, a bad type or a section of the wrong shape must not fail the
    # call at pipeline construction.
    def resolve(turn):
        return resolve_hybrid_turn(
            {"turn": turn}, uses_external_turns=True, is_realtime=False
        )

    assert resolve({"source": "local", "hybrid": {"hold_ms": "99999"}}) == HybridTurn(
        0, 10000
    )
    assert resolve({"source": "local", "hybrid": {"wait_ms": -5}}) == HybridTurn(
        0, 1500
    )
    assert resolve({"source": "local", "hybrid": {"hold_ms": "abc"}}) == HybridTurn(
        0, 1500
    )
    assert resolve({"source": "local", "hybrid": {"hold_ms": None}}) == HybridTurn(
        0, 1500
    )
    assert resolve({"source": "local", "hybrid": "nope"}) == HybridTurn(0, 1500)
    assert resolve(["local"]) is None


def test_source_stt_without_server_turns_behaves_like_auto():
    cfg = {"turn": {"source": "stt"}}
    assert (
        resolve_hybrid_turn(cfg, uses_external_turns=False, is_realtime=False) is None
    )
    # part 2 turns this into a 422; today it is a warning (plan autoauditoría 1)


def test_hybrid_uses_local_strategies_not_external():
    start = _create_non_realtime_user_turn_start_strategies(
        {}, uses_external_turns=False
    )
    stop = _create_non_realtime_user_turn_stop_strategies({}, uses_external_turns=False)
    assert isinstance(start[0], TranscriptionUserTurnStartStrategy)
    assert isinstance(stop[0], SpeechTimeoutUserTurnStopStrategy)
    # Control: with external turns the stock strategies are still the external ones.
    assert isinstance(
        _create_non_realtime_user_turn_start_strategies({}, uses_external_turns=True)[
            0
        ],
        ExternalUserTurnStartStrategy,
    )
    assert isinstance(
        _create_non_realtime_user_turn_stop_strategies({}, uses_external_turns=True)[0],
        ExternalUserTurnStopStrategy,
    )


class _Transport:
    """Stand-ins are real processors: ``Pipeline.__init__`` links the list it is given."""

    def input(self):
        return FrameProcessor(name="in")

    def output(self):
        return FrameProcessor(name="out")


def _named(name: str) -> FrameProcessor:
    return FrameProcessor(name=name)


class _AnswerSupervisor(FrameProcessor):
    """The supervisor is a processor that hands ``build_pipeline`` its context gate."""

    def __init__(self):
        super().__init__(name="answer_supervisor")

    def llm_gate(self):
        return _named("llm_gate")


def _build(absorber, answer_supervisor=None):
    return build_pipeline(
        _Transport(),
        _named("stt"),
        _named("audio"),
        _named("user_agg"),
        _named("assistant_agg"),
        _named("monitor"),
        [_named("generation")],
        _named("metrics"),
        _named("funnel"),
        answer_supervisor=answer_supervisor,
        turn_signal_absorber=absorber,
    )


def _names(pipeline) -> list[str]:
    # ``Pipeline._processors`` is ``[source, *processors, sink]``; the source is the
    # pipeline's own, so the wired list starts at index 1.
    return [
        "absorber" if isinstance(p, TurnSignalAbsorberProcessor) else p.name
        for p in pipeline._processors[1:-1]
    ]


TAIL = ["monitor", "generation", "out", "audio", "assistant_agg", "metrics"]


# Pipecat 1.12 rewrite of "absorber right behind the STT": point A became point B. With an
# answer supervisor (the outbound case) the absorber goes BELOW it, because the supervisor
# times the answer on Flux's raw ``ProposedUserStartedSpeakingFrame``
# (``answer_supervisor.py:261-266``), which the absorber discards; and ABOVE the aggregator,
# whose turn signals it is there to replace.
def test_absorber_sits_at_point_b():
    pipeline = _build(TurnSignalAbsorberProcessor(), _AnswerSupervisor())
    assert _names(pipeline) == [
        "in",
        "funnel",
        "stt",
        "answer_supervisor",
        "absorber",
        "user_agg",
        "llm_gate",
        *TAIL,
    ]


# Rewrite of "no absorber keeps today's order": today's order is now upstream's
# (``stt → answer_supervisor → user_aggregator → llm_gate → call_monitor → generation``).
def test_no_absorber_keeps_todays_order():
    assert _names(_build(None, _AnswerSupervisor())) == [
        "in",
        "funnel",
        "stt",
        "answer_supervisor",
        "user_agg",
        "llm_gate",
        *TAIL,
    ]
    assert _names(_build(None)) == ["in", "funnel", "stt", "user_agg", *TAIL]


# Rewrite of "absorber above the voicemail detector": the merge deleted our
# ``VoicemailDetector`` block for upstream's ``AnswerSupervisor``, which classifies the
# aggregator's turn message instead of frames (``answer_supervisor.py:120-128, :284``), so it
# no longer needs to sit below the absorber. What still holds without a supervisor (inbound):
# the absorber is the last thing before the aggregator.
def test_absorber_sits_right_above_the_user_aggregator_without_a_supervisor():
    pipeline = _build(TurnSignalAbsorberProcessor())
    assert _names(pipeline)[:5] == ["in", "funnel", "stt", "absorber", "user_agg"]
