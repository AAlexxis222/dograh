"""Fix-wave-2 measurements (red-team register M1, M5, M6, §5.3-15). Every test here writes
its cells to ``out/results-B.md``; results-A stays the S0-S11 matrix. M2's cells are written
from the S6/S11 tests in ``test_scenarios.py`` (same file)."""

from dataclasses import replace

import pytest
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState as E

from api.tests.spike_hybrid_turn import scenarios as S
from api.tests.spike_hybrid_turn.harness import Scenario, run_scenario
from api.tests.spike_hybrid_turn.test_scenarios import _instants, _row, lost_total

# The lag past which a Flux final is counted as "clearly after" the local close: the local
# close itself jitters by up to one tolerance around the VAD stop, so a lag inside it is the
# transition band (recorded, not asserted).
LAG_DROP_MS = 50


def _joined(r) -> str:
    return " ".join(m for _, m in r.messages).strip()


def _flux_final_instant(r) -> float:
    ends = [t for t, kind, _ in r.stub_events if kind == "EndOfTurn"]
    assert len(ends) == 1, r.stub_events
    return ends[0]


# --- M1: R2 sensitivity to the local-close / Flux-final lag (S5, F1) ------------------------
# S5's local turn closes at ~2203-2228 (second VAD stop at 2200 + COMPLETE); the catalogue's
# Flux final lands at 2400. Sweep the final over the close and record where the "sábado" tail
# starts being dropped by the orphan rule. 2100 is added to the brief's set so one cell sits
# clearly (> TOLERANCE_MS) before the close and the "tail present" branch is exercised:
# the 2200 cell measured a lag of -5.6 ms, inside the transition band.
LAG_SWEEP_END_AT = [2100, 2200, 2250, 2300, 2400, 2600, 3000]


@pytest.mark.asyncio
@pytest.mark.parametrize("end_at", LAG_SWEEP_END_AT)
async def test_m1_s5_lag_sweep_f1(results_b, end_at):
    sc = replace(
        S.S5,
        id=f"S5-end{end_at}",
        flux=[ev if ev[1] != "end" else (end_at, "end", S.TEXT) for ev in S.S5.flux],
    )
    r = await run_scenario(sc, absorber_mode="f1")
    local_close = _instants(r, "UP:UserStoppedSpeakingFrame")[-1]
    flux_final = _flux_final_instant(r)
    lag = flux_final - local_close
    dropped = r.absorber_stats["orphan_text_dropped"]
    _row(
        results_b,
        sc.id,
        "f1",
        0,
        r,
        note=(
            f"M1 flux_final_at={flux_final:.1f} local_close_at={local_close:.1f} "
            f"lag_ms={lag:+.1f} orphan_text_dropped={dropped}"
        ),
    )
    assert len(r.messages) >= 1, r.events
    if lag < -S.TOLERANCE_MS:
        # Final clearly before the local close: the local turn is open, the tail arrives as
        # the "sábado" delta and the context gets the whole sentence.
        assert _joined(r) == S.TEXT, (lag, r.messages, dict(r.absorber_stats))
        assert dropped == 0, dict(r.absorber_stats)
    elif lag > LAG_DROP_MS:
        # Final clearly after the local close: orphan rule, tail dropped.
        assert _joined(r) == S.PART, (lag, r.messages, dict(r.absorber_stats))
        assert dropped == 1, dict(r.absorber_stats)
        assert lost_total(r) == 1, dict(r.absorber_stats)
    # -TOLERANCE_MS <= lag <= LAG_DROP_MS: transition band, recorded only. Measured
    # 2026-09-10 at end_at=2200 (lag -5.6 ms): the tail was delivered, but as a SECOND
    # message — the delta landed while the local turn was already closing, so it opened a
    # transcript-only turn of its own (ghost=True under the §5.3-15 metric).


# --- M5: stub atomicity between the final and Flux's UserStopped (S1, S5; F1) ---------------
YIELD = {"yield_between_final_and_stop": True}


@pytest.mark.asyncio
async def test_m5_s1_f1_with_yield_between_final_and_stop(results_b):
    sc = replace(S.S1, id="S1-yield", end_kwargs=YIELD)
    r = await run_scenario(sc, absorber_mode="f1")
    _row(
        results_b,
        sc.id,
        "f1",
        0,
        r,
        note="M5 stub yields the loop between final and UserStopped",
    )
    # Same cell as results-A `S1|f1`: Flux's UserStopped is never a barrier (spec §1.1).
    assert len(r.messages) == 1 and r.messages[0][1].strip() == S.TEXT, r.events
    assert r.counts["DOWN:UserStartedSpeakingFrame"] == 0, r.counts
    assert r.absorber_stats["promoted"] == 1, dict(r.absorber_stats)
    assert r.absorber_stats["orphan_avoided"] == 1, dict(r.absorber_stats)
    assert lost_total(r) == 0, dict(r.absorber_stats)


@pytest.mark.asyncio
async def test_m5_s5_f1_with_yield_between_final_and_stop(results_b):
    sc = replace(S.S5, id="S5-yield", end_kwargs=YIELD)
    r = await run_scenario(sc, absorber_mode="f1")
    _row(
        results_b,
        sc.id,
        "f1",
        0,
        r,
        note="M5 stub yields the loop between final and UserStopped",
    )
    # Same cell as results-A `S5|f1`: the final at 2400 is an orphan either way.
    assert len(r.messages) == 1 and r.messages[0][1].strip() == S.PART, r.events
    assert r.absorber_stats["orphan_text_dropped"] == 1, dict(r.absorber_stats)
    assert lost_total(r) == 1, dict(r.absorber_stats)


# --- M6: VAD jitter stress (S1, S5; F1) -----------------------------------------------------
JITTERS = [0, 15, 30, 60]


@pytest.mark.asyncio
@pytest.mark.parametrize("jitter", JITTERS)
@pytest.mark.parametrize(
    "sc,expected", [(S.S1, S.TEXT), (S.S5, S.PART)], ids=["S1", "S5"]
)
async def test_m6_vad_jitter_f1(results_b, sc, expected, jitter):
    r = await run_scenario(sc, absorber_mode="f1", vad_jitter_ms=jitter)
    _row(
        results_b,
        f"{sc.id}-jitter{jitter}",
        "f1",
        0,
        r,
        note=f"M6 vad_jitter_ms={jitter} windows={r.vad_windows}",
    )
    if jitter <= S.TOLERANCE_MS:
        # Inside the declared tolerance the cell must not move.
        assert len(r.messages) == 1, r.events
        assert r.messages[0][1].strip() == expected, (jitter, r.messages)
    # 60 ms (twice the tolerance) is recorded, not asserted.


# --- §5.3-15: a ghost metric that can fire ---------------------------------------------------
# No VAD at all: the final alone opens and closes a local turn (transcript-only fallback,
# stop strategy ``_handle_transcription``), so the analyzer never returns COMPLETE.
SGHOST = Scenario(
    id="Sghost",
    flux=[(0, "start", ""), (1400, "end", S.TEXT)],
    speaking=[],
    verdicts=[E.COMPLETE],
    end_at=8000,
)


@pytest.mark.asyncio
async def test_ghost_metric_fires_on_transcript_only_turn_without_absorber(results_b):
    r = await run_scenario(SGHOST, absorber_mode=None)
    ghost = _row(
        results_b, "Sghost", None, 0, r, note="§5.3-15 final with no local turn"
    )
    assert len(r.messages) == 1, r.events
    assert r.local_turns_completed == 0, r.events
    assert ghost is True, (r.messages, r.local_turns_completed)


@pytest.mark.asyncio
async def test_ghost_metric_with_absorber_on_transcript_only_turn(results_b):
    """MEASURED 2026-09-10, NOT the brief's expectation of ``ghost is False``.

    Spec §3's final rules: with nothing emitted for the Flux turn the final PASSES THROUGH
    (``passthrough_final``), and the absorber only swallows a final when text already went
    out and no local turn is open. Here nothing was ever promoted (no VAD, no interim), so
    the absorber forwards the final and the aggregator opens the same transcript-only turn
    it opens without the absorber: ghost=True in both rows. The absorber can make the flag
    False only where it swallows (S8-shape: results-A ``S8|f1`` messages=1, COMPLETE=1).
    Whether a final with no prior interim and no local turn should be swallowed is a spec-§3
    decision, not something this spike can settle.
    """
    r = await run_scenario(SGHOST, absorber_mode="f1")
    ghost = _row(
        results_b, "Sghost", "f1", 0, r, note="§5.3-15 final with no local turn"
    )
    assert r.local_turns_completed == 0, r.events
    assert r.absorber_stats["passthrough_final"] == 1, dict(r.absorber_stats)
    assert len(r.messages) == 1, r.events
    assert ghost is True, (r.messages, r.local_turns_completed)
