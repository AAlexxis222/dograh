"""Scenario catalogue (spec §4). Times in ms from speech start. VAD windows end at the
instant the local VAD reports stop (stop_secs=0.2 already elapsed)."""
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState as E

from api.tests.spike_hybrid_turn.harness import Scenario

TEXT = "quiero reservar para el sábado"

S0 = Scenario(
    id="S0",
    flux=[(0, "start", ""), (800, "eager", TEXT), (1400, "end", TEXT)],
    speaking=[(0, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
)

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
    flux=[(0, "start", ""), (700, "eager", PART), (1200, "resumed", ""), (2400, "end", TEXT)],
    speaking=[(0, 900), (1200, 2200)],
    verdicts=[E.INCOMPLETE, E.COMPLETE],
    end_at=9000,
)  # local detector holds the turn open across the pause

S5B = Scenario(
    id="S5b",
    flux=[(0, "start", ""), (700, "eager", PART), (1200, "resumed", ""), (2400, "end", TEXT)],
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
    flux=[(0, "start", ""), (300, "eager", "quiero"), (800, "eager", TEXT), (1400, "end", TEXT)],
    speaking=[(60, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
    start="min_words",
    bot_speaking_at=-500,
)  # MinWords over sparse interims: late start and aggregation reset

S11 = Scenario(
    id="S11",
    flux=[(0, "start", ""), (800, "eager", TEXT), (1400, "end", TEXT)],
    speaking=[(800, 1000)],
    verdicts=[E.COMPLETE],
    end_at=8000,
    bot_speaking_at=-500,
)  # interruption and eager interim on the same deadline: queue flush vs text
