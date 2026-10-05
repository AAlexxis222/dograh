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
SREPEAT = Scenario(
    id="Srepeat",
    flux=[
        (0, "start", ""),
        (800, "update", S.TEXT),
        (2000, "update", S.TEXT),  # repeat after the local close, bot talking
        (2400, "end", S.TEXT),
        (2600, "update", S.TEXT),  # repeat after the final
    ],
    speaking=[(0, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
    bot_speaking_at=1300,
)


def _interruptions_after_bot_started(r) -> int:
    keys = [key for _, key in r.events]
    first = keys.index("UP:BotStartedSpeakingFrame")
    return keys[first:].count("UP:InterruptionFrame")


@pytest.mark.asyncio
async def test_repeated_update_interim_after_local_close_opens_no_turn():
    r = await run_scenario(SREPEAT, absorber=True)
    turns_opened = r.counts["UP:UserStartedSpeakingFrame"]
    assert turns_opened == 1, r.events
    assert _interruptions_after_bot_started(r) == 0, r.events
    assert [m[1].strip() for m in r.messages] == [S.TEXT]
