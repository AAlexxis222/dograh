"""Turn analyzer stub with a scripted verdict sequence and the full surface `ta` uses
(spec §3, §11 A13): append_audio, analyze_end_of_turn, update_vad_start_secs, clear, params."""

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.metrics.metrics import MetricsData


class ScriptedAnalyzer(BaseTurnAnalyzer):
    def __init__(self, verdicts: list[EndOfTurnState]):
        super().__init__()
        self._verdicts = list(verdicts)
        self._params = SmartTurnParams(stop_secs=0.2)
        self.calls: list[EndOfTurnState] = []

    @property
    def speech_triggered(self) -> bool:
        return False

    @property
    def params(self) -> SmartTurnParams:
        return self._params

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        state = self._verdicts.pop(0) if self._verdicts else EndOfTurnState.COMPLETE
        self.calls.append(state)
        return state, None

    def update_vad_start_secs(self, vad_start_secs: float) -> None:
        pass

    def clear(self) -> None:
        pass
