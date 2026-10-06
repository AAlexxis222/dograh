"""Pipecat 1.12 turns EmptyUserTurnConfig ON by default (llm_response_universal.py:188).
VOZ-AC-B2-28: keep today's behaviour until a measured decision changes it.
VOZ-BUG-18: the runtime reader resolves legacy/absent turn_start_strategy like the backfill."""

from api.services.pipecat import run_pipeline


def test_user_aggregator_params_disable_empty_user_turn():
    params = run_pipeline.build_user_aggregator_params()
    assert params.empty_user_turn is None


def _start_types(doc):
    strategies = run_pipeline._create_non_realtime_user_turn_start_strategies(
        doc, uses_external_turns=False
    )
    return [type(s) for s in strategies]


def test_legacy_and_absent_turn_start_build_the_same_strategy_at_runtime():
    legacy = _start_types({"turn_start_strategy": "provisional_vad"})
    assert legacy == _start_types({})
    assert [t.__name__ for t in legacy] != ["MinWordsUserTurnStartStrategy"]
