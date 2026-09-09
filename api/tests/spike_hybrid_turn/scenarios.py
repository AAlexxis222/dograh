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
