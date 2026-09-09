"""VAD analyzer driven by the scenario timeline: the real VADController runs, so VAD frames
are born at the right instant with real timestamp/stop_secs (spec §4, §11 A3/B10)."""
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState

from api.tests.spike_hybrid_turn.timeline import Timeline


class ScriptedVAD(VADAnalyzer):
    def __init__(self, timeline: Timeline, windows: list[tuple[int, int]], params: VADParams):
        super().__init__(params=params)
        self._timeline = timeline
        self._windows = windows

    def num_frames_required(self) -> int:
        return 1

    def voice_confidence(self, buffer: bytes) -> float:
        return 1.0

    async def analyze_audio(self, buffer: bytes) -> VADState:
        # Bypass volume gating: MockTransport audio is silence (spec §11 A3).
        if self._timeline.t0 is None:
            return VADState.QUIET
        t = self._timeline.now_ms()
        return VADState.SPEAKING if any(a <= t < b for a, b in self._windows) else VADState.QUIET
