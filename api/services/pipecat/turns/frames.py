"""Frames private to the hybrid turn processors."""

from dataclasses import dataclass

from pipecat.frames.frames import (
    DataFrame,
    TranscriptionFrame,
    UninterruptibleFrame,
)


@dataclass
class TranscriptionReplaceFrame(DataFrame, UninterruptibleFrame):
    """The STT rewrote a transcript the absorber already promoted.

    Handled by ``HybridUserAggregator``: the pending user aggregation is replaced by
    ``text`` instead of being appended to (spec §6.1.1 branch 2, D-11). Not a
    ``TranscriptionFrame`` on purpose — the stock aggregator would concatenate it.

    Uninterruptible because the interim it corrects is already in the aggregation: an
    interruption empties the processing queue of everything else (``FrameQueue.reset``),
    which would leave the superseded text as the user's message. Same reasoning as
    ``FunctionCallResultFrame``.
    """

    text: str
    user_id: str
    timestamp: str

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: [{self.text}])"


@dataclass
class HeldTranscriptionFrame(TranscriptionFrame, UninterruptibleFrame):
    """A final the absorber held back and is now releasing (D-13).

    Uninterruptible because the local turn start that releases it makes the aggregator
    broadcast an interruption a few awaits later (``llm_response_universal.py:1243``),
    and an interruption flushes every queued interruptible frame
    (``FrameProcessor._start_interruption``): the text would be counted as delivered and
    then dropped. Same reasoning as ``TranscriptionReplaceFrame``.

    A plain ``TranscriptionFrame`` to everyone else: the aggregator dispatches on
    ``isinstance`` (``llm_response_universal.py:795``), so nothing else changes.
    """
