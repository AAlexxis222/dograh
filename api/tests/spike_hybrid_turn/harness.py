"""Scenario runner: MockTransport + FluxStub + [Absorber] + real user aggregator (VAD stub
inside) + mock LLM/TTS. One monotonic clock, absolute deadlines, clean shutdown (spec §4)."""

import asyncio
from collections import Counter
from dataclasses import dataclass, field

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests import ContextCapturingMockLLM, MockTTSService
from pipecat.tests.mock_transport import MockTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_mute import (
    FunctionCallUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start import (
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from api.tests.spike_hybrid_turn.analyzer_stub import ScriptedAnalyzer
from api.tests.spike_hybrid_turn.flux_stub import FluxStub
from api.tests.spike_hybrid_turn.timeline import Timeline
from api.tests.spike_hybrid_turn.vad_stub import ScriptedVAD

WATCHED = (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    InterruptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,  # register M2: which interims reach the aggregator, and when
    # The output transport pushes a downstream AND an upstream copy (base_output.py:763-773).
    # This tap sits before the aggregator and after nothing that emits it, so only the UPSTREAM
    # copy crosses it: a barge-in scenario can prove the bot really was talking.
    BotStartedSpeakingFrame,
)


@dataclass
class Scenario:
    id: str
    flux: list[tuple[int, str, str]]  # (t_ms, "start"|"eager"|"resumed"|"end", text)
    speaking: list[tuple[int, int]]  # VAD speech windows [a, b) in ms
    verdicts: list[EndOfTurnState]
    end_at: (
        int  # when to stop the pipeline (>= last event + 6500 for ghost-turn window)
    )
    start: str = "default"  # "default" | "min_words"
    mute_until_bot: bool = False
    bot_speaking_at: int | None = None
    end_kwargs: dict = field(default_factory=dict)  # passed to emit_end_of_turn


@dataclass
class Result:
    messages: list[tuple[float, str]]
    counts: Counter
    absorber_stats: Counter
    transcripts_emitted: int
    transcripts_to_aggregator: int
    dangling_tasks: int
    events: list[tuple[float, str]]
    aggregator: (
        LLMUserAggregator  # live user aggregator, for post-run strategy inspection
    )
    # Wall-clock shift applied to this run (see Timeline). Every t_ms above is already back in
    # scenario coordinates; this is here so a row can say the run was shifted at all.
    offset_ms: int = 0
    # Register M2: DOWN InterimTranscriptionFrames that crossed the tap (i.e. reached the
    # aggregator's input) before / after the first InterruptionFrame crossed it.
    interim_reached_before_interrupt: int = 0
    interim_reached_after_interrupt: int = 0
    # Register M7 / §5.3-16: Flux finals that entered the absorber and left through none of
    # its counted exits (must be 0 unless a frame was flushed inside the absorber).
    dropped_by_absorber: int = 0
    # §5.3-15: COMPLETE decisions the scripted analyzer returned (local turns it closed).
    local_turns_completed: int = 0
    # Stub log (t_ms, kind, text), scenario coordinates: the instant of Flux's final even when
    # the absorber swallowed it (register M1).
    stub_events: list[tuple[float, str, str]] = field(default_factory=list)
    vad_windows: list[tuple[int, int]] = field(
        default_factory=list
    )  # after jitter (M6)


class Tap(FrameProcessor):
    """Counts watched frames by direction and records them on the timeline."""

    def __init__(self, timeline: Timeline, counts: Counter, events: list, **kwargs):
        super().__init__(**kwargs)
        self._timeline, self._counts, self._events = timeline, counts, events
        self._interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, WATCHED):
            key = f"{'DOWN' if direction == FrameDirection.DOWNSTREAM else 'UP'}:{frame.__class__.__name__}"
            self._counts[key] += 1
            if isinstance(frame, InterruptionFrame):
                self._interrupted = True
            elif (
                isinstance(frame, InterimTranscriptionFrame)
                and direction == FrameDirection.DOWNSTREAM
            ):
                self._counts[
                    "interim_after_interrupt"
                    if self._interrupted
                    else "interim_before_interrupt"
                ] += 1
            t = self._timeline.now_ms() if self._timeline.t0 is not None else -1.0
            self._events.append(
                (
                    t,
                    key
                    + (
                        f" '{frame.text}'"
                        if isinstance(
                            frame, (TranscriptionFrame, InterimTranscriptionFrame)
                        )
                        else ""
                    ),
                )
            )
        await self.push_frame(frame, direction)


def _start_strategies(kind: str):
    if kind == "min_words":
        return [MinWordsUserTurnStartStrategy(min_words=3), VADUserTurnStartStrategy()]
    return [TranscriptionUserTurnStartStrategy(), VADUserTurnStartStrategy()]  # rp:189


def _stub_transcripts_pushed(stub: FluxStub, sc: Scenario) -> int:
    """TranscriptionFrames the stub actually pushed downstream.

    Only ``EndOfTurn`` pushes a final, and only when the scenario didn't ask the stub to
    suppress it (min_confidence drop). Counting the pushes — not the events — is what makes
    ``transcripts_emitted - transcripts_to_aggregator`` measure queue loss and nothing else.
    """
    if sc.end_kwargs.get("suppress_transcript"):
        return 0
    return sum(1 for _, kind, _ in stub.emitted if kind == "EndOfTurn")


def _timeline_offset(sc: Scenario) -> int:
    """Wall-clock shift that keeps every scheduled instant of ``sc`` at or after the start.

    Only scheduled instants take part: the VAD windows are read through ``Timeline.now_ms()``,
    which already reports scenario time, so they follow the shift on their own.
    """
    times = [t for t, _, _ in sc.flux] + [sc.end_at]
    if sc.bot_speaking_at is not None:
        times.append(sc.bot_speaking_at)
    return max(0, -min(times))


FINAL_OUTCOMES = (
    "passthrough_final",
    "rewrite",
    "delta_emitted",
    "dup_avoided",
    "orphan_avoided",
)


def _dropped_by_absorber(stats: Counter) -> int:
    """Finals that entered ``_on_flux_final`` minus the ones that took a counted exit."""
    return stats["finals_received"] - sum(stats[k] for k in FINAL_OUTCOMES)


async def run_scenario(
    sc: Scenario,
    *,
    absorber_mode: str | None,
    hybrid_wait_ms: int = 0,
    vad_jitter_ms: int = 0,
) -> Result:
    offset = _timeline_offset(sc)
    timeline = Timeline(offset_ms=offset)
    counts: Counter = Counter()
    events: list[tuple[float, str]] = []
    messages: list[tuple[float, str]] = []

    transport = MockTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_end_silence_secs=0,
        ),
        generate_audio=True,
    )
    stub = FluxStub(timeline)
    analyzer = ScriptedAnalyzer(sc.verdicts)
    strategies = UserTurnStrategies(
        start=_start_strategies(sc.start),
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)],
    )
    mute = [FunctionCallUserMuteStrategy()]
    if sc.mute_until_bot:
        mute.insert(0, MuteUntilFirstBotCompleteUserMuteStrategy())
    vad = ScriptedVAD(
        timeline,
        sc.speaking,
        VADParams(stop_secs=0.2),
        jitter_ms=vad_jitter_ms,
        seed=sc.id,
    )
    user_params = LLMUserAggregatorParams(
        user_turn_strategies=strategies,  # explicit: survives Flux's STTMetadataFrame (agg:957-977)
        user_mute_strategies=mute,
        user_turn_stop_timeout=5.0,  # rp:127 local value
        vad_analyzer=vad,
    )
    context = LLMContext(messages=[{"role": "system", "content": "spike"}])
    pair = LLMContextAggregatorPair(
        context,
        assistant_params=LLMAssistantAggregatorParams(),
        user_params=user_params,
    )
    user_agg, assistant_agg = pair.user(), pair.assistant()

    @user_agg.event_handler("on_user_turn_message_added")
    async def _on_msg(aggregator, message):
        messages.append((timeline.now_ms(), message.content))

    # Count what actually reaches the aggregator's transcription path (loss detector, §11 A6/B6).
    reached = Counter()
    original = user_agg._handle_transcription

    async def _counting(frame):
        reached["n"] += 1
        await original(frame)

    user_agg._handle_transcription = (
        _counting  # throwaway spike: monkeypatch is acceptable
    )

    llm = ContextCapturingMockLLM()
    tts = MockTTSService(mock_audio_duration_ms=3000, frame_delay=0)
    processors = [transport.input(), stub]
    absorber = None
    if absorber_mode is not None:
        from api.tests.spike_hybrid_turn.absorber import Absorber

        absorber = Absorber(mode=absorber_mode, hybrid_wait_ms=hybrid_wait_ms)
        processors.append(absorber)
    processors += [
        Tap(timeline, counts, events),
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ]
    task = PipelineTask(Pipeline(processors))

    for t_ms, kind, text in sc.flux:
        if kind == "start":
            timeline.at(t_ms, stub.emit_start_of_turn)
        elif kind == "eager":
            timeline.at(t_ms, lambda text=text: stub.emit_eager(text))
        elif kind == "resumed":
            timeline.at(t_ms, stub.emit_turn_resumed)
        elif kind == "end":
            timeline.at(
                t_ms, lambda text=text: stub.emit_end_of_turn(text, **sc.end_kwargs)
            )
    if sc.bot_speaking_at is not None:
        timeline.at(
            sc.bot_speaking_at,
            lambda: task.queue_frame(TTSSpeakFrame("bot is talking for a while")),
        )
    timeline.at(sc.end_at, task.stop_when_done)

    before = {t for t in asyncio.all_tasks()}
    runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    try:
        await asyncio.wait_for(
            asyncio.gather(timeline.run(), runner),
            timeout=(sc.end_at + offset) / 1000 + 20,
        )
    finally:
        # gather() does not cancel its siblings when one raises, so a failing timeline would
        # leave this runner alive and it would show up in the NEXT scenario's `before` snapshot.
        if not runner.done():
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
    await asyncio.sleep(0.05)
    dangling = [
        t
        for t in asyncio.all_tasks()
        if t not in before and t is not asyncio.current_task() and not t.done()
    ]

    return Result(
        messages=messages,
        counts=counts,
        absorber_stats=Counter(absorber.stats) if absorber else Counter(),
        # Transcripts that left the STT/absorber stage toward the aggregator. With an absorber
        # that is every TranscriptionFrame it forwarded (promotions and deltas included).
        transcripts_emitted=(
            absorber.stats["forwarded"]
            if absorber
            else _stub_transcripts_pushed(stub, sc)
        ),
        transcripts_to_aggregator=reached["n"],
        dangling_tasks=len(dangling),
        events=events,
        aggregator=user_agg,
        offset_ms=offset,
        interim_reached_before_interrupt=counts["interim_before_interrupt"],
        interim_reached_after_interrupt=counts["interim_after_interrupt"],
        dropped_by_absorber=_dropped_by_absorber(absorber.stats) if absorber else 0,
        local_turns_completed=analyzer.complete_count,
        stub_events=list(stub.emitted),
        vad_windows=vad.windows,
    )
