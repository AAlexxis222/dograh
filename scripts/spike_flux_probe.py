"""Probe the real Deepgram Flux service: paced 20 ms audio in, JSONL of every frame/event out.

Part B of the 8a hybrid-turn viability spike. Part A (api/tests/spike_hybrid_turn) drives a
FluxStub that reproduces Flux's frame-emission call pattern; this script feeds the same kind of
audio to the real service so the stub's script can be checked against measured timings.

Usage:
    DEEPGRAM_API_KEY=... python scripts/spike_flux_probe.py clip.wav out.jsonl [--eager 0.5]

Exit codes:
    0  connected and Flux sent at least one StartOfTurn
    1  bad input (unreadable WAV, wrong audio format) or an unexpected crash
    2  connection failed
    3  connected, but Flux never sent StartOfTurn
    4  Flux's watchdog fired: our pacing broke, so the timings cannot be trusted
    5  the pipeline did not drain after the audio ended

argparse also exits 2 on a usage error. The two are told apart by the output file: a connection
failure always leaves a JSONL carrying `connection_error` or `CONNECT_FAILED`, a usage error
never creates one.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import wave

from loguru import logger
from pipecat.frames.frames import (
    BotSpeakingFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    UserSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.transcriptions.language import Language
from pipecat.workers.runner import WorkerRunner

SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000  # PCM16 mono
SILENCE_CHUNK = b"\x00" * CHUNK_BYTES

CONNECT_TIMEOUT_S = 10.0
DRAIN_TIMEOUT_S = 30.0
SHUTDOWN_TIMEOUT_S = 5.0

# Production configuration this probe has to match to be worth anything
# (service_factory.py:508-519 and :538-543).
MODEL = "flux-general-multi"
EOT_THRESHOLD = 0.7
# `language_hints` is a LIST of `Language` (flux/base.py:141); the registry entry for "es" is
# `Language.ES` (service_factory.py:204-215), wrapped in a list at service_factory.py:519.
LANGUAGE_HINTS = [Language.ES]

# The five turn events Flux registers (flux/base.py:223-227). `add_event_handler` silently
# no-ops on an unregistered name (base_object.py:203-206), so these must match exactly.
TURN_EVENTS = (
    "on_start_of_turn",
    "on_eager_end_of_turn",
    "on_end_of_turn",
    "on_update",
    "on_turn_resumed",
)

# The WARNING Flux logs when audio stops flowing mid-turn (flux/base.py:355-357).
WATCHDOG_MARKER = "Sending silence to Flux"

# The high-frequency frames pipecat's own FrameLogger drops (processors/logger.py:34-40).
NOISY_FRAMES = (
    BotSpeakingFrame,
    UserSpeakingFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
)

T0 = time.monotonic()


def now_ms() -> float:
    """Milliseconds since the process started. One clock for the whole run."""
    return (time.monotonic() - T0) * 1000.0


class Jsonl:
    """Append-only JSONL sink. Flushes per record so a crash still leaves the evidence."""

    def __init__(self, path: str):
        self._f = open(path, "w", encoding="utf-8")
        self.n: dict[str, int] = {}

    def write(self, kind: str, **data) -> None:
        self.n[kind] = self.n.get(kind, 0) + 1
        record = {"t_ms": round(now_ms(), 1), "kind": kind, **data}
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class FrameTap(FrameProcessor):
    """Records every frame crossing one point of the pipeline, tagged with which point."""

    def __init__(self, out: Jsonl, tap: str, **kwargs):
        super().__init__(**kwargs)
        self._out = out
        self._tap = tap

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            self._out.write(
                "ErrorFrame",
                tap=self._tap,
                error=str(frame.error),
                fatal=frame.fatal,
                direction=direction.name,
            )
        elif not isinstance(frame, NOISY_FRAMES):
            self._out.write(
                frame.__class__.__name__,
                tap=self._tap,
                text=getattr(frame, "text", None),
                finalized=getattr(frame, "finalized", None),
                direction=direction.name,
            )
        await self.push_frame(frame, direction)


class WatchdogGuard:
    """Catches Flux's watchdog WARNING, which means we stopped feeding audio in time."""

    def __init__(self):
        self.fired = False
        logger.add(self._sink, level="WARNING")

    def _sink(self, message) -> None:
        if WATCHDOG_MARKER in str(message):
            self.fired = True


def load_pcm(path: str) -> bytes:
    """Read a 16 kHz mono PCM16 WAV. Rejects anything else before a connection is opened."""
    with wave.open(path, "rb") as w:
        actual = (w.getframerate(), w.getnchannels(), w.getsampwidth())
        if actual != (SAMPLE_RATE, 1, 2):
            raise SystemExit(
                f"{path}: need 16 kHz mono PCM16, got {actual[0]} Hz, "
                f"{actual[1]} channel(s), {actual[2] * 8}-bit"
            )
        return w.readframes(w.getnframes())


async def feed(task: PipelineWorker, pcm: bytes, silence_ms: int) -> None:
    """Push the clip plus a silence tail at real time, 20 ms per chunk.

    Deadlines are absolute against a single start instant, so a slow chunk does not push every
    later chunk back: without that, drift alone would trip Flux's watchdog and invalidate the
    measured gaps.
    """
    chunks = [pcm[i : i + CHUNK_BYTES] for i in range(0, len(pcm), CHUNK_BYTES)]
    chunks += [SILENCE_CHUNK] * (silence_ms // CHUNK_MS)
    start = time.monotonic()
    for i, buf in enumerate(chunks):
        delay = start + i * (CHUNK_MS / 1000.0) - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        await task.queue_frame(
            InputAudioRawFrame(audio=buf, sample_rate=SAMPLE_RATE, num_channels=1)
        )


def turn_event_handler(out: Jsonl, name: str):
    """Build the handler for one Flux turn event. Handlers are called as (service, *args)."""

    async def handler(service, *args):
        out.write(name, text=args[0] if args else None)

    return handler


async def await_connection(connected: asyncio.Event, failed: asyncio.Event) -> bool:
    """Wait for whichever of the two connection outcomes happens first."""
    waiters = [asyncio.create_task(e.wait()) for e in (connected, failed)]
    _, pending = await asyncio.wait(
        waiters, timeout=CONNECT_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED
    )
    for waiter in pending:
        waiter.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)
    return connected.is_set()


async def shutdown(task: PipelineWorker, runner: asyncio.Task) -> None:
    """Stop the pipeline and reap the runner, so no task outlives the process."""
    await task.cancel()
    done, _ = await asyncio.wait([runner], timeout=SHUTDOWN_TIMEOUT_S)
    if not done:
        runner.cancel()
    await asyncio.gather(runner, return_exceptions=True)


async def probe(args: argparse.Namespace, api_key: str, pcm: bytes) -> int:
    out = Jsonl(args.out)
    try:
        guard = WatchdogGuard()
        # `reconnect_on_error` is not passed: the service already hardcodes it False
        # (flux/stt.py:237), and an unknown kwarg is swallowed in silence by
        # BaseObject.__init__, so passing it would only look like it did something.
        stt = DeepgramFluxSTTService(
            api_key=api_key,
            should_interrupt=False,  # service_factory.py:541
            sample_rate=SAMPLE_RATE,
            settings=DeepgramFluxSTTService.Settings(
                model=MODEL,
                eot_threshold=EOT_THRESHOLD,
                eager_eot_threshold=args.eager,
                eot_timeout_ms=args.eot_timeout_ms,
                keyterm=[],
                language_hints=LANGUAGE_HINTS,
            ),
        )

        connected = asyncio.Event()
        failed = asyncio.Event()

        @stt.event_handler("on_connected")  # registered in stt_service.py:188
        async def _on_connected(service):
            out.write("connected")
            connected.set()

        @stt.event_handler("on_connection_error")  # registered in stt_service.py:190
        async def _on_connection_error(service, error):
            out.write("connection_error", error=str(error))
            failed.set()

        for event in TURN_EVENTS:
            stt.add_event_handler(event, turn_event_handler(out, event))

        pipeline = Pipeline([FrameTap(out, "up"), stt, FrameTap(out, "down")])
        task = PipelineWorker(
            pipeline, params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE)
        )
        runner = asyncio.create_task(WorkerRunner(handle_sigint=False).run(task))

        if not await await_connection(connected, failed):
            out.write("CONNECT_FAILED")
            await shutdown(task, runner)
            return 2

        await feed(task, pcm, silence_ms=args.eot_timeout_ms + 500)
        await task.stop_when_done()
        try:
            await asyncio.wait_for(runner, timeout=DRAIN_TIMEOUT_S)
        except TimeoutError:
            out.write("DRAIN_TIMEOUT")
            await shutdown(task, runner)
            return 5

        if guard.fired:
            return 4
        if out.n.get("on_start_of_turn", 0) == 0:
            return 3
        print(json.dumps(out.n, ensure_ascii=False))
        return 0
    finally:
        out.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", help="16 kHz mono PCM16 clip to send")
    parser.add_argument("out", help="JSONL file to write the frame/event log to")
    parser.add_argument("--eager", type=float, default=0.5, help="eager_eot_threshold")
    parser.add_argument(
        "--eot-timeout-ms",
        type=int,
        default=3000,  # service_factory.py:510
        help="eot_timeout_ms; the silence tail is this plus 500 ms",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("DEEPGRAM_API_KEY missing (gate, spec section 9)", file=sys.stderr)
        return 2
    pcm = load_pcm(args.wav)
    return asyncio.run(probe(args, api_key, pcm))


if __name__ == "__main__":
    sys.exit(main())
