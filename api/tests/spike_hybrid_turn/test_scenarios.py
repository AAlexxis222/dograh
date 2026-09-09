"""S0-S11 (spec §4). S0 is the control run WITHOUT absorber and is not mutation-tested."""
import pytest

from api.tests.spike_hybrid_turn import scenarios as S
from api.tests.spike_hybrid_turn.harness import run_scenario


def _row(results, sc_id, mode, wait, r):
    ghost = any(t > 6000 for t, _ in r.messages)
    results.add(
        scenario=sc_id, mode=mode or "none", wait_ms=wait, messages=len(r.messages),
        texts=[m for _, m in r.messages], down_started=r.counts["DOWN:UserStartedSpeakingFrame"],
        down_stopped=r.counts["DOWN:UserStoppedSpeakingFrame"],
        interruptions=r.counts["DOWN:InterruptionFrame"] + r.counts["UP:InterruptionFrame"],
        lost=r.transcripts_emitted - r.transcripts_to_aggregator, ghost=ghost, stats=dict(r.absorber_stats),
    )
    assert r.dangling_tasks == 0, f"dangling tasks after {sc_id}: {r.dangling_tasks}"
    return ghost


@pytest.mark.asyncio
async def test_s0_control_records_baseline(results):
    """Control: aligned turn, no absorber. Records the baseline every later run is read against.

    Measured 2026-09-09 (see the task report): Flux's DOWN UserStarted/UserStopped and the
    aggregator's UP copies each appear exactly once at this tap, the local VAD frames are born
    at the scripted instants (1.5 ms / 1000.9 ms for the [0, 1000) window), and exactly one
    message lands at ~1407 ms with the full text. The pathology the brief predicted here
    (doubled signals at the tap, or a close blocked by ctl:365) does NOT reproduce in the
    aligned case: the local VAD stop clears ``_user_speaking`` at 1000.9 ms, well before the
    final transcript triggers the close. The tap sits upstream of the aggregator, so the
    duplication that reaches the LLM/TTS is out of its view by construction.
    """
    r = await run_scenario(S.S0, absorber_mode=None)
    _row(results, "S0", None, 0, r)
    assert len(r.messages) >= 1, (r.counts, r.events)
    assert r.dangling_tasks == 0


@pytest.mark.asyncio
async def test_local_strategies_survive_stt_metadata():
    """Flux recommends ``ExternalUserTurnStrategies``; explicit local strategies win (agg:957-977).

    Also the second consecutive S0 run on the session loop: proves nothing leaks between
    scenarios (spec §4 lifecycle).
    """
    r = await run_scenario(S.S0, absorber_mode=None)

    # Non-vacuous: the metadata frame really landed. The field defaults to None and only
    # _handle_stt_metadata sets it; Flux announces ttfs_p99_latency=0.0 (spec §1.1).
    assert r.aggregator._ttfs_p99_latency == 0.0

    live = r.aggregator._user_turn_controller.user_turn_strategies
    start_names = [type(s).__name__ for s in live.start]
    stop_names = [type(s).__name__ for s in live.stop]
    assert stop_names == ["TurnAnalyzerUserTurnStopStrategy"], stop_names
    assert start_names == ["TranscriptionUserTurnStartStrategy", "VADUserTurnStartStrategy"], start_names
    assert r.dangling_tasks == 0
