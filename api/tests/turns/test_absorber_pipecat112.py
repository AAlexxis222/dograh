"""The absorber on Pipecat 1.12's Flux sequence (VOZ-G0-08, spec B2 VOZ-AC-B2-12 reseat):
``ProposedUserStarted`` → interims on ``Update`` → [``EagerTranscriptionFrame``] → final on
``EndOfTurn`` → ``ProposedUserStopped`` (``flux/stt_base.py:816, :960, :906-918, :921``)."""

import pytest
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState as E
from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)

from api.tests.turns import scenarios as S
from api.tests.turns.fakes import run_absorber
from api.tests.turns.harness import Scenario, run_scenario

PROPOSED = (ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame)


@pytest.mark.asyncio
async def test_proposed_frames_never_reach_the_aggregator():
    eager = EagerTranscriptionFrame("hola", "", "")
    out, _ = await run_absorber(
        [
            ProposedUserStartedSpeakingFrame(),
            InterimTranscriptionFrame("hola", "", ""),
            eager,
            ProposedUserStoppedSpeakingFrame(),
            TranscriptionFrame("hola", "", ""),
        ]
    )
    assert not [f for f in out if isinstance(f, PROPOSED)]
    # Tolerated, not acted on: hybrid runs with speculation off, so an eager transcript that
    # still arrives goes on untouched (the rewrite that counts it is VOZ-FASE-4).
    assert eager in out


@pytest.mark.asyncio
async def test_proposed_frames_dropped_also_while_muted():
    out, _ = await run_absorber([ProposedUserStartedSpeakingFrame()], user_muted=True)
    assert out == []


# B2 adversarial log #2 (CRITICAL): the local turn closes on the promoted interim and the bot
# answers; Flux keeps sending ``Update`` with the same words (in the silence before its
# ``EndOfTurn``, and once more after it). In hybrid the start strategy reads interims
# (``TranscriptionUserTurnStartStrategy(use_interim=True)``), so each repeat that reached the
# aggregator would open a turn and cut the bot.
def _late_update_scenario(id_: str, *late_texts: str) -> Scenario:
    """``late_texts`` arrive in order after the local close (bot talking), and again after the final."""
    return Scenario(
        id=id_,
        flux=[
            (0, "start", ""),
            (800, "update", S.TEXT),
            *[(2000 + 150 * i, "update", t) for i, t in enumerate(late_texts)],
            (2400, "end", S.TEXT),
            *[(2600 + 150 * i, "update", t) for i, t in enumerate(late_texts)],
        ],
        speaking=[(0, 1000)],
        verdicts=[E.COMPLETE],
        end_at=8000,
        bot_speaking_at=1300,
    )


# Flux's late ``Update`` may also rewrite the last word or drop one (Opus review G0 #2): with
# the local turn closed only an interim that adds words may reach the aggregator.
LATE_UPDATES = [
    _late_update_scenario("Srepeat", S.TEXT),
    _late_update_scenario("Srewritten", "quiero reservar para el domingo"),
    _late_update_scenario("Sshorter", "quiero reservar para el"),
    # Shorter, then the dropped word restored: still nothing beyond the longest text seen.
    _late_update_scenario("Sshorter_restored", "quiero reservar para el", S.TEXT),
]


def _interruptions_after_bot_started(r) -> int:
    keys = [key for _, key in r.events]
    first = keys.index("UP:BotStartedSpeakingFrame")
    return keys[first:].count("UP:InterruptionFrame")


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", LATE_UPDATES, ids=lambda sc: sc.id)
async def test_late_update_interim_after_local_close_opens_no_turn(scenario):
    r = await run_scenario(scenario, absorber=True)
    turns_opened = r.counts["UP:UserStartedSpeakingFrame"]
    assert turns_opened == 1, r.events
    assert _interruptions_after_bot_started(r) == 0, r.events
    assert [m[1].strip() for m in r.messages] == [S.TEXT]


@pytest.mark.asyncio
async def test_late_interim_with_new_words_still_reaches_the_aggregator():
    """The filter drops only what adds nothing: words beyond the mark are real speech."""
    scenario = _late_update_scenario("Sextended", S.TEXT + " por la tarde")
    r = await run_scenario(scenario, absorber=True)
    assert r.counts["UP:UserStartedSpeakingFrame"] == 2, r.events


# The shorter ``Update`` arrives while the local turn is still open (forwarded), and the word
# comes back after the close: the mark is the longest text seen, not the last one forwarded.
SHORTER_WHILE_OPEN = Scenario(
    id="Sshorter_while_open",
    flux=[
        (0, "start", ""),
        (700, "update", S.TEXT),
        (850, "update", "quiero reservar para el"),
        (2000, "update", S.TEXT),
        (2400, "end", S.TEXT),
        (2600, "update", S.TEXT),
    ],
    speaking=[(0, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
    bot_speaking_at=1300,
)


@pytest.mark.asyncio
async def test_mark_is_the_longest_text_not_the_last_forwarded():
    r = await run_scenario(SHORTER_WHILE_OPEN, absorber=True)
    keys = [key for _, key in r.events]
    closed = keys.index("UP:UserStoppedSpeakingFrame")
    # The restored word is not new: no interim reaches the aggregator after the close.
    # (The final's tail beyond the promoted text is the final path's business, not this filter's.)
    assert not [
        k for k in keys[closed:] if k.startswith("DOWN:InterimTranscriptionFrame")
    ], r.events


# Review r1 F2: the late filter is for interims after a local turn CLOSED. Before the local turn
# opens (production min_words: MinWords(3) alone, no VAD) a Flux revision must not freeze the
# interims that follow, or MinWords never sees 3 words and the barge-in waits for the final.
REVISION_BEFORE_OPEN = Scenario(
    id="Srevision_before_open",
    flux=[
        (100, "start", ""),
        (300, "update", "eh espera"),
        (500, "update", "espera un"),  # Flux revises the first word
        (700, "update", "espera un momento"),
        (900, "update", "espera un momento por"),
        (2500, "end", "espera un momento por favor"),
    ],
    speaking=[(100, 2300)],
    verdicts=[E.COMPLETE],
    end_at=9000,
    start="min_words_only",
    bot_speaking_at=0,
)


@pytest.mark.asyncio
async def test_revision_before_the_local_turn_opens_does_not_delay_the_barge_in():
    r = await run_scenario(REVISION_BEFORE_OPEN, absorber=True)
    interruptions = [
        t
        for t, key in r.events
        if key == "UP:InterruptionFrame" and t > REVISION_BEFORE_OPEN.bot_speaking_at
    ]
    assert interruptions and interruptions[0] < 1000, (interruptions, r.absorber_stats)
