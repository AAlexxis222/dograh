"""Per-clip metrics over the JSONL that ``spike_flux_probe.py`` writes (8a spike, §5).

For every JSONL given, and for every Flux turn in it (a turn starts at ``on_start_of_turn``),
print:

    gap_ms     EagerEndOfTurn -> EndOfTurn (last eager event to the end event), or "-"
    relation   how the final relates to the LAST eager text:
               extension (final starts with the eager text), rewrite (it does not), no_eager
    updates    number of ``on_update`` events in the turn
    resumed    number of ``on_turn_resumed`` events in the turn

Usage:
    python scripts/spike_flux_probe_metrics.py out/clip-a-0.5.jsonl [more.jsonl ...]
    python scripts/spike_flux_probe_metrics.py --selftest

Record shape (``Jsonl.write`` in spike_flux_probe.py): one JSON object per line with
``t_ms`` (float, ms since process start), ``kind`` (event or frame class name) and, for turn
events, ``text``. Only the five turn-event kinds are read here; frame records are ignored.
No network, no pipecat import.
"""

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field

START, EAGER, END, UPDATE, RESUMED = (
    "on_start_of_turn",
    "on_eager_end_of_turn",
    "on_end_of_turn",
    "on_update",
    "on_turn_resumed",
)


@dataclass
class Turn:
    start_ms: float
    eagers: list[tuple[float, str]] = field(default_factory=list)
    end: tuple[float, str] | None = None
    updates: int = 0
    resumed: int = 0

    @property
    def gap_ms(self) -> float | None:
        if self.end is None or not self.eagers:
            return None
        return round(self.end[0] - self.eagers[-1][0], 1)

    @property
    def relation(self) -> str:
        if not self.eagers:
            return "no_eager"
        if self.end is None:
            return "no_final"
        final, eager = self.end[1] or "", self.eagers[-1][1] or ""
        return "extension" if final.strip().startswith(eager.strip()) else "rewrite"


def parse_records(lines: Iterable[str]) -> list[dict]:
    records = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"line {n}: not JSON ({e})") from e
    return records


def turns_of(records: list[dict]) -> list[Turn]:
    """Group turn events into turns. Events before the first StartOfTurn open an implicit
    turn at their own instant, so a truncated log still yields numbers."""
    turns: list[Turn] = []
    for rec in records:
        kind, t = rec.get("kind"), float(rec.get("t_ms", 0.0))
        if kind == START:
            turns.append(Turn(start_ms=t))
            continue
        if kind not in (EAGER, END, UPDATE, RESUMED):
            continue
        if not turns:
            turns.append(Turn(start_ms=t))
        turn = turns[-1]
        if kind == EAGER:
            turn.eagers.append((t, rec.get("text") or ""))
        elif kind == END:
            turn.end = (t, rec.get("text") or "")
        elif kind == UPDATE:
            turn.updates += 1
        else:
            turn.resumed += 1
    return turns


def format_report(name: str, turns: list[Turn]) -> str:
    lines = [f"{name}: {len(turns)} turn(s)"]
    for i, turn in enumerate(turns, 1):
        gap = "-" if turn.gap_ms is None else f"{turn.gap_ms:.1f}"
        lines.append(
            f"  turn {i} start={turn.start_ms:.1f}ms gap_ms={gap} relation={turn.relation} "
            f"eagers={len(turn.eagers)} updates={turn.updates} resumed={turn.resumed}"
        )
    return "\n".join(lines)


def selftest() -> int:
    """Three synthetic clips: extension, rewrite, no eager. Asserts the numbers."""

    def rec(t, kind, text=None):
        return json.dumps({"t_ms": t, "kind": kind, "text": text})

    extension = [
        rec(0.0, "connected"),
        rec(10.0, START),
        rec(700.0, EAGER, "quiero reservar para el"),
        rec(900.0, UPDATE, "quiero reservar para el"),
        rec(1200.0, RESUMED),
        rec(2400.0, END, "quiero reservar para el sábado"),
        rec(2401.0, "TranscriptionFrame", "quiero reservar para el sábado"),
    ]
    rewrite = [
        rec(10.0, START),
        rec(300.0, EAGER, "quiero"),
        rec(800.0, EAGER, "ola quiero"),
        rec(1400.0, END, "hola quiero reservar"),
    ]
    no_eager = [
        rec(10.0, START),
        rec(500.0, UPDATE, "quiero"),
        rec(1400.0, END, "quiero reservar"),
    ]
    two_turns = [
        rec(10.0, START),
        rec(700.0, EAGER, "hola"),
        rec(1000.0, END, "hola"),
        rec(3000.0, START),
        rec(3200.0, END, "adiós"),
    ]

    (t,) = turns_of(parse_records(extension))
    assert t.gap_ms == 1700.0, t
    assert t.relation == "extension", t
    assert (t.updates, t.resumed, len(t.eagers)) == (1, 1, 1), t

    (t,) = turns_of(parse_records(rewrite))
    assert t.gap_ms == 600.0, t  # from the LAST eager
    assert t.relation == "rewrite", t
    assert (t.updates, t.resumed, len(t.eagers)) == (0, 0, 2), t

    (t,) = turns_of(parse_records(no_eager))
    assert t.gap_ms is None, t
    assert t.relation == "no_eager", t
    assert (t.updates, t.resumed) == (1, 0), t

    a, b = turns_of(parse_records(two_turns))
    assert (a.gap_ms, a.relation) == (300.0, "extension"), a
    assert (b.gap_ms, b.relation) == (None, "no_eager"), b

    print(format_report("selftest/extension", turns_of(parse_records(extension))))
    print(format_report("selftest/rewrite", turns_of(parse_records(rewrite))))
    print(format_report("selftest/no_eager", turns_of(parse_records(no_eager))))
    print("selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "jsonl", nargs="*", help="probe JSONL file(s), one per clip run"
    )
    parser.add_argument("--selftest", action="store_true", help="run on synthetic data")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.jsonl:
        parser.error("give at least one JSONL file, or --selftest")
    for path in args.jsonl:
        with open(path, encoding="utf-8") as f:
            print(format_report(path, turns_of(parse_records(f))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
