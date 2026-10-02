"""Scenario catalogue (spec §4). Times in ms from speech start. VAD windows end at the
instant the local VAD reports stop (stop_secs=0.2 already elapsed)."""

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState as E

from api.tests.turns.harness import Scenario

TEXT = "quiero reservar para el sábado"

# --- R1 double close: Flux's final lands after the local VAD stop (1000 ms) ------------------
S1 = Scenario(
    id="S1",
    flux=[(0, "start", ""), (800, "eager", TEXT), (1400, "end", TEXT)],
    speaking=[(0, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
)

S2 = Scenario(
    id="S2",
    flux=[(0, "start", ""), (800, "eager", TEXT), (900, "end", TEXT)],
    speaking=[(0, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
)

S3 = Scenario(
    id="S3",
    flux=[(0, "start", ""), (800, "eager", TEXT), (1000, "end", TEXT)],
    speaking=[(0, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
)  # same deadline: coincidence

S4 = Scenario(
    id="S4",
    flux=[(0, "start", ""), (1400, "end", TEXT)],
    speaking=[(0, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
)  # no eager

S8 = Scenario(
    id="S8",
    flux=[(0, "start", ""), (800, "eager", TEXT), (1900, "end", TEXT)],
    speaking=[(0, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8500,
)  # orphan final

# --- R2 partial interim promoted while the sentence is still unfinished ---------------------
# The user pauses mid-sentence: Flux emits an EagerEndOfTurn with only the first half, then
# TurnResumed, and the final with the whole sentence lands one second later.
PART = "quiero reservar para el"

S5 = Scenario(
    id="S5",
    flux=[
        (0, "start", ""),
        (700, "eager", PART),
        (1200, "resumed", ""),
        (2400, "end", TEXT),
    ],
    speaking=[(0, 900), (1200, 2200)],
    verdicts=[E.INCOMPLETE, E.COMPLETE],
    end_at=9000,
)  # local detector holds the turn open across the pause

S5B = Scenario(
    id="S5b",
    flux=[
        (0, "start", ""),
        (700, "eager", PART),
        (1200, "resumed", ""),
        (2400, "end", TEXT),
    ],
    speaking=[(0, 900), (1200, 2200)],
    verdicts=[E.COMPLETE, E.COMPLETE],
    end_at=9000,
)  # detector fails at the pause: measures the damage

# --- R3 barge-in: the user talks over the bot -----------------------------------------------
S6 = Scenario(
    id="S6",
    flux=[(0, "start", ""), (800, "eager", TEXT), (1400, "end", TEXT)],
    speaking=[(60, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
    bot_speaking_at=-500,
)  # who raises the interruption, and when

S7 = Scenario(
    id="S7",
    flux=[
        (0, "start", ""),
        (300, "eager", "quiero"),
        (800, "eager", TEXT),
        (1400, "end", TEXT),
    ],
    speaking=[(60, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
    start="min_words",
    bot_speaking_at=-500,
)  # MinWords over sparse interims: late start and aggregation reset

# --- R5 mute: MuteUntilFirstBotComplete over a muted turn, then a normal one ----------------
# Turn 1 (0-1400) happens while the user is muted; the bot speaks at 3000 (3000 ms of mock audio)
# and its completion lifts the mute at ~6540. Turn 2 is scheduled ~2.5 s after that drain, so it
# runs unmuted and must behave like S1 — spec §4's S9, second half ("after the mute, S1 holds").
S9B = Scenario(
    id="S9b",
    flux=[
        (0, "start", ""),
        (800, "eager", TEXT),
        (1400, "end", TEXT),
        (9000, "start", ""),
        (9800, "eager", TEXT),
        (10400, "end", TEXT),
    ],
    speaking=[(0, 1000), (9000, 10000)],
    verdicts=[E.COMPLETE, E.COMPLETE],
    end_at=17000,
    mute_until_bot=True,
    bot_speaking_at=3000,
)

S11 = Scenario(
    id="S11",
    flux=[(0, "start", ""), (800, "eager", TEXT), (1400, "end", TEXT)],
    speaking=[(800, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
    bot_speaking_at=-500,
)  # interruption and eager interim on the same deadline: queue flush vs text

# --- D-13: a final with no interim and no local turn (the local VAD never fired) ------------
SGHOST = Scenario(
    id="Sghost",
    flux=[(0, "start", ""), (1400, "end", TEXT)],
    speaking=[],
    verdicts=[],
    end_at=8000,
)

# --- D-11: Flux rewrites the promoted interim in its final, local turn still open -----------
# S5's timeline with a final that rewrites the promoted interim instead of extending it. The
# second speech window outlives the final (2600 > 2400) on purpose: D-11 is the branch where the
# promoted text is still pending in an OPEN local turn, so it can be replaced. Measured with
# S5's window (…, 2200) the analyzer closes the turn 200 ms before the final arrives and the run
# takes the D-12 orphan branch instead, which cannot retract the message already emitted.
REWRITE_FINAL = "Quiero cancelar para el sábado."
SREWRITE = Scenario(
    id="Srewrite",
    flux=[
        (0, "start", ""),
        (700, "eager", PART),
        (1200, "resumed", ""),
        (2400, "end", REWRITE_FINAL),
    ],
    speaking=[(0, 900), (1200, 2600)],
    verdicts=[E.INCOMPLETE, E.COMPLETE],
    end_at=9000,
)
