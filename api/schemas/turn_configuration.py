"""``workflow_configurations.turn`` — turn-detection controls (spec §6).

Part 1 ships ``source`` and the hybrid knobs. ``source=local`` with a server-turn STT
(Deepgram Flux, Dograh-Flux, Cartesia ink-2) enables the hybrid: Flux transcribes, the
local analyzer decides (``TurnSignalAbsorberProcessor``). ``auto`` keeps today's behaviour
untouched; ``stt`` is accepted now and only differs from ``auto`` once part 2 lands.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_HYBRID_WAIT_MS = 0  # D-10: F1, promote on the local VAD stop, no wait
MAX_HYBRID_WAIT_MS = 2000
DEFAULT_HYBRID_HOLD_MS = (
    1500  # spec §15 (Fable, revisable): hold a final that has no local turn
)
MAX_HYBRID_HOLD_MS = 10000


class HybridTurnConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wait_ms: int = Field(
        default=DEFAULT_HYBRID_WAIT_MS,
        ge=0,
        le=MAX_HYBRID_WAIT_MS,
        description=(
            "Milliseconds to wait for the STT's own end-of-turn after the local VAD stop "
            "before promoting the last interim. 0 promotes immediately (recommended)."
        ),
    )
    hold_ms: int = Field(
        default=DEFAULT_HYBRID_HOLD_MS,
        ge=0,
        le=MAX_HYBRID_HOLD_MS,
        description=(
            "How long a final transcript that arrives with no local turn open is held for "
            "the next local turn before being delivered as a message of its own."
        ),
    )


class TurnConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["auto", "stt", "local"] = "auto"
    hybrid: HybridTurnConfiguration = Field(default_factory=HybridTurnConfiguration)
