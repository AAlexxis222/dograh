"""turn.source=local with a server-turn STT wires the hybrid: local strategies, the absorber
right behind the STT, the hybrid aggregator. Everything else is exactly today's pipeline
(spec §1.3 'expose ≠ change')."""

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


def _build(absorber):
    return build_pipeline(
        _Transport(),
        _named("stt"),
        _named("audio"),
        _named("llm"),
        _named("tts"),
        _named("user_agg"),
        _named("assistant_agg"),
        _named("callbacks"),
        _named("metrics"),
        _named("funnel"),
        turn_signal_absorber=absorber,
    )


def _names(pipeline) -> list[str]:
    # ``Pipeline._processors`` is ``[source, *processors, sink]``; the source is the
    # pipeline's own, so the wired list starts at index 1.
    return [
        "absorber" if isinstance(p, TurnSignalAbsorberProcessor) else p.name
        for p in pipeline._processors[1:-1]
    ]


def test_absorber_sits_right_behind_the_stt():
    pipeline = _build(TurnSignalAbsorberProcessor())
    assert _names(pipeline)[:4] == ["in", "funnel", "stt", "absorber"]


def test_no_absorber_keeps_todays_order():
    assert _names(_build(None))[:4] == ["in", "funnel", "stt", "user_agg"]
