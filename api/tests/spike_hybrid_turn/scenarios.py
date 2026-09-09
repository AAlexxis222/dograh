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
