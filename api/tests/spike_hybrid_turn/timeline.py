"""Single monotonic clock per scenario with absolute deadlines (spec §4 'Reloj')."""
import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass(order=True)
class _Event:
    t_ms: int
    seq: int
    action: Callable[[], Awaitable[None]] = field(compare=False)


class Timeline:
    """Scenario clock. ``offset_ms`` shifts wall time, never scenario time.

    A scenario may schedule something BEFORE its reference instant (``bot_speaking_at=-500``:
    the bot was already talking 500 ms before the speech). Nothing can run before the pipeline
    starts, so the whole scenario is pushed ``offset_ms`` later on the wall clock. Both
    directions of the conversion live here — ``at()`` shifts forward, ``now_ms()`` shifts back —
    so every reported instant (Tap events, message times, stub log, VAD windows) stays in
    scenario coordinates, where 0 is still the reference instant and -500 is really -500.
    """

    def __init__(self, offset_ms: int = 0) -> None:
        self.t0: float | None = None
        self.offset_ms = offset_ms
        self.started = asyncio.Event()
        self._events: list[_Event] = []
        self._seq = 0

    def start(self) -> None:
        self.t0 = time.monotonic()
        self.started.set()

    def now_ms(self) -> float:
        assert self.t0 is not None, "timeline not started"
        return (time.monotonic() - self.t0) * 1000.0 - self.offset_ms

    def at(self, t_ms: int, action: Callable[[], Awaitable[None]]) -> None:
        """Schedule ``action`` at scenario instant ``t_ms`` (may be negative, see offset_ms)."""
        self._seq += 1
        self._events.append(_Event(t_ms, self._seq, action))

    async def run(self) -> None:
        await self.started.wait()
        for ev in sorted(self._events):
            # Absolute deadline: no drift accumulation across events (spec §11 B11).
            delay = self.t0 + (ev.t_ms + self.offset_ms) / 1000.0 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            await ev.action()
