"""Frames private to the hybrid turn processors."""

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame


@dataclass
class TranscriptionReplaceFrame(DataFrame):
    """The STT rewrote a transcript the absorber already promoted.

    Handled by ``HybridUserAggregator``: the pending user aggregation is replaced by
    ``text`` instead of being appended to (spec §6.1.1 branch 2, D-11). Not a
    ``TranscriptionFrame`` on purpose — the stock aggregator would concatenate it.
    """

    text: str
    user_id: str
    timestamp: str

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: [{self.text}])"
