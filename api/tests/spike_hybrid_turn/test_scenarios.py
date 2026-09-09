"""S0-S11 (spec §4). S0 is the control run WITHOUT absorber and is not mutation-tested."""
import pytest

from api.tests.spike_hybrid_turn import scenarios as S
from api.tests.spike_hybrid_turn.harness import run_scenario


def _clusters(times, tol_ms=30.0):
    """Group instants that belong to one logical event.

    The tap sits before the aggregator, so it records only the UPSTREAM copy of an interruption
    (measured: one instant per interruption). This guards the case where a frame is seen more
    than once a few ms apart: fixed 10 ms buckets would split a pair that straddles a boundary,
    so events are clustered by distance instead and the count stays "how many interruptions
    happened", which is the R5 measurement.
    """
    out = []
    for t in sorted(times):
        if not out or t - out[-1][-1] > tol_ms:
            out.append([t])
        else:
            out[-1].append(t)
    return out


def _bot_started(r):
    """Instants the bot was heard starting to speak, in scenario time (empty if it never did)."""
    return [t for t, k in r.events if k.endswith("BotStartedSpeakingFrame")]


def _row(results, sc_id, mode, wait, r):
    ghost = any(t > 6000 for t, _ in r.messages)
    results.add(
        scenario=sc_id, mode=mode or "none", wait_ms=wait, offset=r.offset_ms,
        messages=len(r.messages),
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
    # Floor pinned to the measured baseline: none of these forces the predicted pathology, they
    # only fail if the aligned no-absorber case regresses away from what was measured.
    assert len(r.messages) == 1, (r.counts, r.events)
    assert r.messages[0][1].strip() == S.TEXT, r.messages
    assert r.transcripts_emitted - r.transcripts_to_aggregator == 0, r.events
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


MODES = [("f1", 0), ("f2", 300), ("f2", 600), ("f2", 900)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wait", MODES)
@pytest.mark.parametrize("sc", [S.S1, S.S2, S.S3, S.S8], ids=lambda s: s.id)
async def test_one_message_no_ghost(results, sc, mode, wait):
    r = await run_scenario(sc, absorber_mode=mode, hybrid_wait_ms=wait)
    ghost = _row(results, sc.id, mode, wait, r)
    assert len(r.messages) == 1, r.events
    assert r.messages[0][1].strip() == S.TEXT, r.messages
    assert not ghost
    assert r.counts["DOWN:UserStartedSpeakingFrame"] == 0  # Flux's signals swallowed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wait", MODES)
async def test_s4_no_interim_records_who_closes(results, mode, wait):
    r = await run_scenario(S.S4, absorber_mode=mode, hybrid_wait_ms=wait)
    _row(results, "S4", mode, wait, r)
    # Measured, not prescribed: exactly one message (Flux's final passes through, spec §3) and
    # its arrival time tells whether ta:281 or the 5 s watchdog closed the turn.
    assert len(r.messages) == 1, r.events
    assert r.messages[0][1].strip() == S.TEXT


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wait", MODES)
async def test_s5_pause_mid_sentence_measures_dropped_tail(results, mode, wait):
    """MEASURED 2026-09-09, NOT the brief's expectation of one message with the full sentence.

    The local detector holds the turn open across the pause (INCOMPLETE at the first VAD stop),
    so the only text the absorber can promote is Flux's stale partial — Flux sends no second
    interim after TurnResumed. The second VAD stop closes the local turn at ~2220, ~180 ms
    BEFORE Flux's final arrives at 2400, and the orphan rule (spec §3) then drops the tail
    rather than push text into no open turn: one message with PART and
    ``orphan_text_dropped == 1``, identically in F1 and in all three F2 waits.

    R2's answer for this shape is therefore neither "delta" nor "rewrite" but "dropped". The
    no-absorber row of the same scenario delivers the whole sentence at ~2410, so the loss is
    the absorber's ghost-turn trade-off, not Flux's or the aggregator's.
    """
    r = await run_scenario(S.S5, absorber_mode=mode, hybrid_wait_ms=wait)
    _row(results, "S5", mode, wait, r)
    assert len(r.messages) == 1, r.events
    text = r.messages[0][1].strip()
    assert text == S.PART, r.messages  # a prefix of the sentence: never duplicated, never rewritten
    assert r.absorber_stats["orphan_text_dropped"] == 1, dict(r.absorber_stats)
    assert r.transcripts_emitted - r.transcripts_to_aggregator == 0, r.events


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wait", MODES)
async def test_s5b_detector_fails_measures_damage(results, mode, wait):
    """MEASURED 2026-09-09: the damage depends on the F2 wait, so the brief's content assertion
    (the whole sentence always reaches the context) is relaxed to the two outcomes observed.

    The detector closes the turn at the pause, and the turn analyzer finalizes ~1 ms after a
    transcript reaches the aggregator — so the promotion instant decides:

    * F1 (promote at ~930) and F2 wait=300 (promote at ~1220): the turn closes on the partial,
      a second local turn opens with the resumed speech, and Flux's final is delivered as the
      ``sábado`` delta. Two messages, whole sentence, in order.
    * F2 wait=600/900 (promote at ~1510/~1810): the user is speaking again by then, so the stop
      is deferred (ctl:355) and the turn now closes at ~2220 — before the final at 2400, which
      the orphan rule drops. One message, tail lost, exactly as in S5.

    F2 wait=300 is the racy cell: the timer fires at ~1208 and the resumed speech at ~1200.
    """
    r = await run_scenario(S.S5B, absorber_mode=mode, hybrid_wait_ms=wait)
    _row(results, "S5b", mode, wait, r)
    joined = " ".join(m for _, m in r.messages).strip()
    # Content assertion (spec §11 B8), pinned per cell to the measured outcome instead of the
    # disjunction: only f2/300 is racy, so only it may land on either side.
    if mode == "f1":
        assert joined == S.TEXT, r.messages  # promote at ~930 → close on the partial, then delta
    elif wait >= 600:
        assert joined == S.PART, r.messages  # promote lands mid-resumed-speech → tail orphaned
    else:
        assert joined in (S.TEXT, S.PART), r.messages  # f2/300: timer ~1208 vs speech ~1200
    assert r.transcripts_emitted - r.transcripts_to_aggregator == 0, r.events


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wait", MODES)
async def test_s6_bargein_one_interruption_from_local_vad(results, mode, wait):
    """Barge-in: the user talks over a bot that started 500 ms earlier (R3/R5).

    Measured 2026-09-09: exactly one interruption per run, raised by the local VAD start at
    62-89 ms — Flux's own StartOfTurn is swallowed and its interim at 800 never gets to raise
    one, so barge-in latency is the VAD's, not the STT's. Only the aggregator's UPSTREAM
    broadcast crosses this tap (the downstream copy is queued inside the aggregator, which sits
    after it), hence one recorded instant per interruption. No text is lost in the flush.
    """
    r = await run_scenario(S.S6, absorber_mode=mode, hybrid_wait_ms=wait)
    _row(results, "S6", mode, wait, r)
    ints = [t for t, k in r.events if k.endswith("InterruptionFrame")]
    bot = _bot_started(r)
    # There IS a bot to barge in on: an interruption is broadcast on every user-turn start
    # (agg:1240-1241), so without this the scenario would prove nothing about barge-in.
    assert bot and min(bot) < min(ints), (bot, ints)
    assert len(_clusters(ints)) == 1, r.events  # one interruption event
    assert min(ints) < 300, "interruption must come from the local VAD start, not wait for the interim"
    assert r.transcripts_emitted - r.transcripts_to_aggregator == 0, "text lost in interruption flush"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wait", MODES)
async def test_s7_min_words_measures_late_start_and_reset(results, mode, wait):
    """MinWords(3) over sparse interims, bot speaking (R3).

    Measured 2026-09-09: the aggregation reset never fires and S7 is indistinguishable from S6
    in all four modes. ``VADUserTurnStartStrategy`` opens the turn at ~62-89 ms, and turn start
    calls ``handle_user_turn_started`` on EVERY start strategy (ctl:322-326), which clears
    MinWords' ``_bot_speaking``; from then on its threshold is 1 word, so the 1-word interim at
    300 triggers (a no-op on an open turn) instead of calling ``trigger_reset_aggregation``.
    MinWords only erases aggregated text where no VAD start strategy runs beside it.
    """
    r = await run_scenario(S.S7, absorber_mode=mode, hybrid_wait_ms=wait)
    _row(results, "S7", mode, wait, r)
    ints = [t for t, k in r.events if k.endswith("InterruptionFrame")]
    bot = _bot_started(r)
    assert bot and min(bot) < min(ints), (bot, ints)  # the bot really was talking (min_words=3 armed)
    assert len(r.messages) >= 1
    assert r.messages[-1][1].strip().endswith("sábado"), r.messages  # reset by 1-word interim must not lose the tail


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wait", MODES)
async def test_s11_interruption_overlapping_eager_keeps_text(results, mode, wait):
    """Eager interim and the interruption on the same deadline (R3, queue flush).

    Measured 2026-09-09: the text survives in all four modes. The absorber registers the interim
    ~5-10 ms before the InterruptionFrame comes back upstream, so ``interim_frame`` is set and
    the re-emission fires (``reemitted_interim == 1`` everywhere). The architectural window the
    Task 3 review predicted — the interim still in the absorber's OWN input queue when
    ``_start_interruption`` flushes it (fp:871-890), leaving nothing to re-emit — does not
    reproduce at this timing; the flush arrives after the interim was processed.
    """
    r = await run_scenario(S.S11, absorber_mode=mode, hybrid_wait_ms=wait)
    _row(results, "S11", mode, wait, r)
    ints = [t for t, k in r.events if k.endswith("InterruptionFrame")]
    bot = _bot_started(r)
    assert bot and min(bot) < min(ints), (bot, ints)  # the flush is a real barge-in flush
    # The flush window itself, not just the text: at wait >= 600 the final passes through and
    # would carry the sentence even if the re-emit had found nothing to re-emit.
    assert r.absorber_stats["reemitted_interim"] == 1, dict(r.absorber_stats)
    assert any(S.TEXT in m for _, m in r.messages), r.events


@pytest.mark.asyncio
@pytest.mark.parametrize("sc", [S.S5, S.S5B, S.S6, S.S7, S.S11], ids=lambda s: s.id)
async def test_mutation_without_absorber_fails_r2_r3(results, sc):
    """The absorber is load-bearing for every R2/R3 scenario: without it Flux's DOWN turn signals
    reach the aggregator in all five (S5/S5b additionally show the absorber's own cost — the
    no-absorber run keeps the tail those scenarios' absorbed runs drop)."""
    r = await run_scenario(sc, absorber_mode=None)
    _row(results, sc.id, None, 0, r)
    ok = len(r.messages) == 1 and r.messages[0][1].strip() == S.TEXT and r.counts["DOWN:UserStartedSpeakingFrame"] == 0
    assert not ok, "absorber is not load-bearing for this scenario"


@pytest.mark.asyncio
@pytest.mark.parametrize("sc", [S.S1, S.S8], ids=lambda s: s.id)
async def test_mutation_without_absorber_fails_r1(results, sc):
    r = await run_scenario(sc, absorber_mode=None)
    _row(results, sc.id, None, 0, r)
    assert not (len(r.messages) == 1 and not any(t > 6000 for t, _ in r.messages)
                and r.counts["DOWN:UserStartedSpeakingFrame"] == 0), "absorber is not load-bearing"
