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
    def __init__(self) -> None:
        self.t0: float | None = None
        self.started = asyncio.Event()
        self._events: list[_Event] = []
        self._seq = 0

    def start(self) -> None:
        self.t0 = time.monotonic()
        self.started.set()

    def now_ms(self) -> float:
        assert self.t0 is not None, "timeline not started"
        return (time.monotonic() - self.t0) * 1000.0

    def at(self, t_ms: int, action: Callable[[], Awaitable[None]]) -> None:
        self._seq += 1
        self._events.append(_Event(t_ms, self._seq, action))

    async def run(self) -> None:
        await self.started.wait()
        for ev in sorted(self._events):
            # Absolute deadline: no drift accumulation across events (spec §11 B11).
            delay = self.t0 + ev.t_ms / 1000.0 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            await ev.action()
