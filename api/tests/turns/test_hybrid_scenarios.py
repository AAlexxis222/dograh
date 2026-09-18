"""S1-S11 of the 8a spike replayed against the production absorber, with the production
expectations of spec §6.1.1 (ratified form): one message with the full text in the aligned
scenarios, the whole sentence in S5 (orphan tail emitted), no ghost turn in Sghost, no
duplicated context in Srewrite, no dangling tasks, and Flux's turn signals never reach the
aggregator. Slow (~2-3 min): run in the foreground with a 600 s timeout."""

import pytest

from api.tests.turns import scenarios as S
from api.tests.turns.harness import run_scenario


def _ghost(r) -> bool:
    return len(r.messages) > r.local_turns_completed


def _lost_total(r) -> int:
    return (r.transcripts_emitted - r.transcripts_to_aggregator) + r.dropped_by_absorber


def _texts(r):
    return [m[1].strip() for m in r.messages]


@pytest.mark.asyncio
@pytest.mark.parametrize("sc", [S.S1, S.S2, S.S3, S.S4, S.S8], ids=lambda s: s.id)
async def test_aligned_scenarios_one_message_full_text(sc):
    r = await run_scenario(sc, absorber=True)
    assert _texts(r) == [S.TEXT], r.events
    assert not _ghost(r)
    assert r.counts["DOWN:UserStartedSpeakingFrame"] == 0, r.counts
    assert r.counts["DOWN:UserStoppedSpeakingFrame"] == 0, r.counts
    assert _lost_total(r) == 0
    assert r.dangling_tasks == 0


@pytest.mark.asyncio
async def test_s5_pause_mid_sentence_delivers_the_whole_sentence():
    # D-12: the local turn closes ~190 ms before Flux's final, so the tail is nobody's. The
    # spike dropped it (PART only, `orphan_text_dropped=1`); production leaves it as a message
    # of its own, which is what the orphan branch below pins. Loss is checked by `_lost_total`.
    r = await run_scenario(S.S5, absorber=True)
    assert " ".join(_texts(r)) == S.TEXT, r.events
    assert r.absorber_stats["orphan_emitted"] == 1, r.absorber_stats
    assert _lost_total(r) == 0
    assert r.dangling_tasks == 0


@pytest.mark.asyncio
async def test_s5b_detector_fails_at_the_pause_still_delivers_everything():
    r = await run_scenario(S.S5B, absorber=True)
    assert " ".join(_texts(r)) == S.TEXT, r.events
    assert _lost_total(r) == 0


@pytest.mark.asyncio
async def test_srewrite_replaces_instead_of_duplicating():
    r = await run_scenario(S.SREWRITE, absorber=True)
    joined = " ".join(_texts(r))
    assert "reservar" not in joined, r.events  # the promoted partial was replaced
    assert S.REWRITE_FINAL in joined
    assert (
        r.absorber_stats["rewrite_replaced"]
        + r.absorber_stats["orphan_rewrite_emitted"]
        == 1
    )
    assert r.dangling_tasks == 0


@pytest.mark.asyncio
async def test_sghost_no_local_turn_is_held_then_delivered_without_ghost_flag_lying():
    r = await run_scenario(S.SGHOST, absorber=True, hold_ms=500)
    assert _texts(r) == [S.TEXT]
    assert r.absorber_stats["final_held_no_local_turn"] == 1
    assert r.absorber_stats["held_released_as_message"] == 1
    # The message exists but no local analyzer closed it: the metric must say so.
    assert _ghost(r)


@pytest.mark.asyncio
@pytest.mark.parametrize("sc", [S.S6, S.S11], ids=lambda s: s.id)
async def test_bargein_one_interruption_from_local_vad(sc):
    r = await run_scenario(sc, absorber=True)
    assert _texts(r) == [S.TEXT]
    assert r.counts["UP:InterruptionFrame"] >= 1
    assert r.counts["DOWN:UserStartedSpeakingFrame"] == 0


@pytest.mark.asyncio
async def test_s7_min_words_still_one_message():
    r = await run_scenario(S.S7, absorber=True)
    assert _texts(r) == [S.TEXT]
    assert r.counts["DOWN:UserStartedSpeakingFrame"] == 0


@pytest.mark.asyncio
async def test_s9_muted_turn_then_normal_turn():
    r = await run_scenario(S.S9B, absorber=True)
    # Turn 1 is muted (MuteUntilFirstBotComplete); turn 2 behaves like S1. One message total:
    # the muted turn leaked nothing, not even the final the absorber held and released.
    assert len(_texts(r)) == 1, r.messages
    assert _texts(r)[-1] == S.TEXT, r.events
    # §5.3-4: under mute the turn signals are passed through instead of swallowed. WHICH of the
    # two crosses is a race the scenario cannot fix: the aggregator broadcasts its mute at
    # pipeline start and Flux's StartOfTurn is scheduled at t=0, so the start signal may be
    # swallowed before the mute lands (measured: the stop signal is the one passed through).
    assert r.absorber_stats["passthrough_muted_signal"] >= 1, r.absorber_stats
    assert (
        r.counts["DOWN:UserStartedSpeakingFrame"]
        + r.counts["DOWN:UserStoppedSpeakingFrame"]
    ) >= 1, r.counts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sc", [S.S1, S.S5, S.S6, S.S7, S.S8, S.S11], ids=lambda s: s.id
)
async def test_mutation_without_absorber_breaks(sc):
    r = await run_scenario(sc, absorber=False)
    broken = (
        r.counts["DOWN:UserStartedSpeakingFrame"] > 0
        or _texts(r) != [S.TEXT]
        or _ghost(r)
    )
    assert broken, r.events
