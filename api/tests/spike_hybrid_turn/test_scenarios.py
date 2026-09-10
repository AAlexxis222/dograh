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


def _row(results, sc_id, mode, wait, r, *, ghost_window_ms: int | None = 6000):
    """Record one cell of the results table.

    ``ghost_window_ms`` is the instant past which a message can only be a ghost turn: every
    single-turn scenario ends its speech long before 6000, so anything later is text pushed
    into a turn nobody opened. ``None`` disables the flag for scenarios that legitimately speak
    past that instant (S9's second turn), where a late message proves nothing either way.
    """
    ghost = ghost_window_ms is not None and any(
        t > ghost_window_ms for t, _ in r.messages
    )
    results.add(
        scenario=sc_id,
        mode=mode or "none",
        wait_ms=wait,
        offset=r.offset_ms,
        messages=len(r.messages),
        texts=[m for _, m in r.messages],
        down_started=r.counts["DOWN:UserStartedSpeakingFrame"],
        down_stopped=r.counts["DOWN:UserStoppedSpeakingFrame"],
        interruptions=r.counts["DOWN:InterruptionFrame"]
        + r.counts["UP:InterruptionFrame"],
        lost=r.transcripts_emitted - r.transcripts_to_aggregator,
        ghost=ghost,
        stats=dict(r.absorber_stats),
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
    assert start_names == [
        "TranscriptionUserTurnStartStrategy",
        "VADUserTurnStartStrategy",
    ], start_names
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
    assert text == S.PART, (
        r.messages
    )  # a prefix of the sentence: never duplicated, never rewritten
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
        assert joined == S.TEXT, (
            r.messages
        )  # promote at ~930 → close on the partial, then delta
    elif wait >= 600:
        assert joined == S.PART, (
            r.messages
        )  # promote lands mid-resumed-speech → tail orphaned
    else:
        assert joined in (S.TEXT, S.PART), (
            r.messages
        )  # f2/300: timer ~1208 vs speech ~1200
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
    assert min(ints) < 300, (
        "interruption must come from the local VAD start, not wait for the interim"
    )
    assert r.transcripts_emitted - r.transcripts_to_aggregator == 0, (
        "text lost in interruption flush"
    )


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
    assert bot and min(bot) < min(ints), (
        bot,
        ints,
    )  # the bot really was talking (min_words=3 armed)
    assert len(r.messages) >= 1
    assert r.messages[-1][1].strip().endswith("sábado"), (
        r.messages
    )  # reset by 1-word interim must not lose the tail


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
    assert bot and min(bot) < min(ints), (
        bot,
        ints,
    )  # the flush is a real barge-in flush
    # The flush window itself, not just the text: at wait >= 600 the final passes through and
    # would carry the sentence even if the re-emit had found nothing to re-emit.
    assert r.absorber_stats["reemitted_interim"] == 1, dict(r.absorber_stats)
    assert any(S.TEXT in m for _, m in r.messages), r.events


def _instants(r, key_prefix):
    """Scenario instants of every tapped event whose key starts with ``key_prefix``."""
    return [t for t, k in r.events if k.startswith(key_prefix)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wait", MODES)
async def test_s9_mute_then_normal_turn(results, mode, wait):
    """MEASURED 2026-09-10, NOT the brief's expectation of one message from the second turn.

    ``MuteUntilFirstBotComplete`` mutes from the first non-lifecycle frame and only unmutes on
    ``BotStoppedSpeakingFrame``. The bot's TTS is queued at 3000, starts at ~3020 and its 3000 ms
    of mock audio finishes draining at ~6540 — about 130-160 ms AFTER the second turn's final at
    ~6405. So BOTH turns are muted end to end and the context stays empty: the aggregator
    suppresses every user frame while muted (agg:1085-1097), including the two finals the
    absorber forwarded, which is what the ``lost == 2`` cell of this row means (suppression, not
    queue loss). The scenario measures the reset table under mute, not a post-mute turn.

    The absorber's own behaviour is exactly spec §3's reset table:

    * It stops promoting (``promoted == 0``) — ``_on_local_vad_stop`` returns early while muted,
      so neither the F1 promotion nor any F2 timer fires at either VAD stop. Mutation-checked
      2026-09-10: this cell is over-determined. Removing the mute gate from the promotion path
      still yields ``promoted == 0``, because the aggregator broadcasts no upstream
      ``UserStartedSpeakingFrame`` while muted and the local-turn-open gate alone blocks it. So
      S9 pins the outcome but does NOT exercise the mute branch of ``_promote``; the branch
      that IS discriminating here is the swallow one (see the next bullet, which does fall when
      mutated: ``passthrough_muted_signal`` 3 -> 0).
    * It stops swallowing: 3 of Flux's 4 turn signals are FORWARDED
      (``passthrough_muted_signal``). The 4th is the very first StartOfTurn, swallowed because
      it is itself the frame whose arrival at the aggregator flips the mute on — the upstream
      ``UserMuteStartedFrame`` cannot reach the absorber before the absorber has already handled
      the frame that caused it. Structural, not racy: measured at 0.3 ms vs 1.1 ms.
    """
    r = await run_scenario(S.S9, absorber_mode=mode, hybrid_wait_ms=wait)
    _row(results, "S9", mode, wait, r, ghost_window_ms=None)
    assert (
        r.counts["UP:UserMuteStartedFrame"] == 1
        and r.counts["UP:UserMuteStoppedFrame"] == 1
    ), r.counts
    # The mute really outlives the whole scenario's speech — without this the empty context
    # below would prove nothing about mute (margin measured at 126-160 ms across 5 runs).
    mute_stopped = _instants(r, "UP:UserMuteStoppedFrame")
    assert mute_stopped[0] > _instants(r, "DOWN:TranscriptionFrame")[-1], r.events
    assert r.messages == [], r.messages
    # Reset table: no promotion while muted, and every one of Flux's 4 turn signals accounted
    # for — 3 forwarded, 1 swallowed before the mute could be known (see docstring).
    assert r.absorber_stats["promoted"] == 0, dict(r.absorber_stats)
    assert r.absorber_stats["passthrough_muted_signal"] == 3, dict(r.absorber_stats)
    assert r.absorber_stats["swallowed_UserStartedSpeakingFrame"] == 1, dict(
        r.absorber_stats
    )
    assert r.transcripts_emitted - r.transcripts_to_aggregator == 2, r.events


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wait", MODES)
async def test_s9b_post_mute_turn_behaves_like_s1(results, mode, wait):
    """Spec §4's other half of S9: after the mute lifts, S1 holds.

    The second turn is scheduled at 9000, ~2.5 s after the unmute at ~6530, so it runs fully
    unmuted. Measured 2026-09-10: it reproduces S1 cell for cell — one message with the whole
    sentence in all four modes, F1 and F2/300 promoting the interim and then dropping the
    final's duplicate (``promoted: 1`` + ``orphan_avoided: 1``), F2/600 and F2/900 letting the
    final through instead (``passthrough_final``). Message lands at ~10014 (f1), ~10341
    (f2/300), ~10406 (f2/600), ~10412 (f2/900).

    Two cells differ from the brief's prediction and are pinned to the measurement:

    * ``down_started == 0``, not 1. The mute window covers only turn 1's UserStopped; turn 1's
      StartOfTurn precedes the mute (structural, see ``test_s9_mute_then_normal_turn``) and
      turn 2's arrives long after it lifted, so BOTH starts are swallowed normally and only one
      signal is forwarded (``passthrough_muted_signal: 1``).
    * ``lost == 1``, not 0. The second turn loses nothing; the 1 is turn 1's muted final, which
      the absorber forwarded and the aggregator suppressed. Identical in the no-absorber run,
      so it measures the mute, not the absorber.
    """
    r = await run_scenario(S.S9B, absorber_mode=mode, hybrid_wait_ms=wait)
    _row(results, "S9b", mode, wait, r, ghost_window_ms=None)
    assert (
        r.counts["UP:UserMuteStartedFrame"] >= 1
        and r.counts["UP:UserMuteStoppedFrame"] >= 1
    ), r.counts
    # The second turn really is post-mute: the unmute precedes every event it produced (~2.5 s
    # of margin). Without this the S1-like result below would not be attributable to the unmute.
    mute_stopped = _instants(r, "UP:UserMuteStoppedFrame")
    assert mute_stopped[0] < min(t for t, _ in r.events if t > 8000), r.events
    late = [m for t, m in r.messages if t > 8000]
    assert len(r.messages) == 1 and len(late) == 1, r.messages
    assert late[0].strip() == S.TEXT, r.messages
    assert r.absorber_stats["promoted"] <= 1, dict(r.absorber_stats)
    # Turn 2 loses nothing; the 1 is turn 1's muted final (see docstring).
    assert r.transcripts_emitted - r.transcripts_to_aggregator == 1, r.events
    assert r.counts["DOWN:UserStartedSpeakingFrame"] == 0, r.counts
    assert r.absorber_stats["passthrough_muted_signal"] == 1, dict(r.absorber_stats)


@pytest.mark.asyncio
@pytest.mark.parametrize("sc", [S.S5, S.S5B, S.S6, S.S7, S.S11], ids=lambda s: s.id)
async def test_mutation_without_absorber_fails_r2_r3(results, sc):
    """The absorber is load-bearing for every R2/R3 scenario: without it Flux's DOWN turn signals
    reach the aggregator in all five (S5/S5b additionally show the absorber's own cost — the
    no-absorber run keeps the tail those scenarios' absorbed runs drop)."""
    r = await run_scenario(sc, absorber_mode=None)
    _row(results, sc.id, None, 0, r)
    ok = (
        len(r.messages) == 1
        and r.messages[0][1].strip() == S.TEXT
        and r.counts["DOWN:UserStartedSpeakingFrame"] == 0
    )
    assert not ok, "absorber is not load-bearing for this scenario"
    # Falsifiable form of the same claim: Flux's StartOfTurn really reaches the aggregator.
    assert r.counts["DOWN:UserStartedSpeakingFrame"] >= 1, r.counts
    if sc in (S.S5, S.S5B):
        # The load-bearing R2 measurement: without the absorber the "sábado" tail is NOT
        # dropped — the whole sentence lands in one message (results-A `S5|none`, `S5b|none`).
        assert len(r.messages) == 1, r.messages
        assert r.messages[0][1].strip() == S.TEXT, r.messages


@pytest.mark.asyncio
async def test_mutation_without_absorber_s9_is_mute_governed(results):
    """MEASURED: S9 is the one scenario where the absorber is NOT load-bearing for the context.

    The mute governs the message axis, so removing the absorber changes nothing there — the
    context is empty either way. What the mutation does change is Flux's downstream turn
    signals: with the absorber only 1 of the 2 StartOfTurn frames reaches the aggregator (the
    pre-mute one, see ``test_s9_mute_then_normal_turn``), without it both do. That difference is
    the whole of the absorber's contribution under mute, and it is what R5's reset table asks
    for: while muted the absorber steps aside and lets the aggregator do the suppressing.
    """
    r = await run_scenario(S.S9, absorber_mode=None)
    _row(results, "S9", None, 0, r, ghost_window_ms=None)
    assert r.counts["DOWN:UserStartedSpeakingFrame"] == 2, r.counts  # absorbed cells: 1
    assert r.messages == [], r.messages  # identical to every absorbed cell


@pytest.mark.asyncio
async def test_mutation_without_absorber_fails_s9b(results):
    """S9B's post-mute turn IS absorber-governed on the turn-signal axis, not on the text axis.

    Without the absorber both of Flux's StartOfTurn frames reach the aggregator (2 vs 0 in every
    absorbed cell). The message is delivered either way — the aggregator's own path handles this
    aligned turn — so S9B pins where the absorber earns its place after the mute lifts: it is
    back to swallowing, exactly as in S1.
    """
    r = await run_scenario(S.S9B, absorber_mode=None)
    _row(results, "S9b", None, 0, r, ghost_window_ms=None)
    late = [m for t, m in r.messages if t > 8000]
    ok = (
        r.counts["DOWN:UserStartedSpeakingFrame"] == 0
        and len(late) == 1
        and late[0].strip() == S.TEXT
    )
    assert not ok, "absorber is not load-bearing for S9B"
    assert r.counts["DOWN:UserStartedSpeakingFrame"] == 2, r.counts  # absorbed cells: 0
    assert len(late) == 1 and late[0].strip() == S.TEXT, (
        r.messages
    )  # text axis: unchanged


@pytest.mark.asyncio
@pytest.mark.parametrize("sc", [S.S1, S.S2, S.S3, S.S4, S.S8], ids=lambda s: s.id)
async def test_mutation_without_absorber_fails_r1(results, sc):
    r = await run_scenario(sc, absorber_mode=None)
    _row(results, sc.id, None, 0, r)
    assert not (
        len(r.messages) == 1
        and not any(t > 6000 for t, _ in r.messages)
        and r.counts["DOWN:UserStartedSpeakingFrame"] == 0
    ), "absorber is not load-bearing"
    # Falsifiable form of the same claim: Flux's StartOfTurn really reaches the aggregator.
    assert r.counts["DOWN:UserStartedSpeakingFrame"] >= 1, r.counts
