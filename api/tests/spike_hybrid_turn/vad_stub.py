"""VAD analyzer driven by the scenario timeline: the real VADController runs, so VAD frames
are born at the right instant with real timestamp/stop_secs (spec §4, §11 A3/B10)."""

import random

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState

from api.tests.spike_hybrid_turn.timeline import Timeline


def jittered_windows(
    windows: list[tuple[int, int]], jitter_ms: int, seed: str
) -> list[tuple[int, int]]:
    """Shift every scripted instant by a deterministic offset in [-jitter_ms, +jitter_ms].

    Seeded by the scenario id so a cell reproduces run to run (red team F17 / register M6:
    a real VAD does not report on the scripted millisecond).
    """
    if jitter_ms <= 0:
        return list(windows)
    rng = random.Random(seed)
    return [
        (a + rng.randint(-jitter_ms, jitter_ms), b + rng.randint(-jitter_ms, jitter_ms))
        for a, b in windows
    ]


class ScriptedVAD(VADAnalyzer):
    def __init__(
        self,
        timeline: Timeline,
        windows: list[tuple[int, int]],
        params: VADParams,
        *,
        jitter_ms: int = 0,
        seed: str = "",
    ):
        super().__init__(params=params)
        self._timeline = timeline
        self._windows = jittered_windows(windows, jitter_ms, seed)

    @property
    def windows(self) -> list[tuple[int, int]]:
        """The windows actually used (jitter applied), for the results row."""
        return self._windows

    def num_frames_required(self) -> int:
        return 1

    def voice_confidence(self, buffer: bytes) -> float:
        return 1.0

    async def analyze_audio(self, buffer: bytes) -> VADState:
        # Bypass volume gating: MockTransport audio is silence (spec §11 A3).
        if self._timeline.t0 is None:
            return VADState.QUIET
        t = self._timeline.now_ms()
        return (
            VADState.SPEAKING
            if any(a <= t < b for a, b in self._windows)
            else VADState.QUIET
        )
