"""Session-scoped results table for the scenario runs (spec §4)."""

from pathlib import Path

import pytest

OUT = Path(__file__).parent / "out"
COLUMNS = [
    "scenario",
    "mode",
    "wait_ms",
    "offset",
    "messages",
    "texts",
    "down_started",
    "down_stopped",
    "interruptions",
    "lost",
    "lost_total",  # lost + orphan_text_dropped + dropped_by_absorber (register M7)
    "ghost",
    "stats",
    "note",  # per-cell measured extras (results-B); empty in the S0-S11 matrix
]


class Results:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, **row):
        self.rows.append(row)

    def write(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True)
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        for row in self.rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in COLUMNS) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def results():
    """S0-S11 matrix (``results-A.md``)."""
    r = Results()
    yield r
    r.write(OUT / "results-A.md")


@pytest.fixture(scope="session")
def results_b():
    """Fix-wave-2 measurements (``results-B.md``), same columns as results-A."""
    r = Results()
    yield r
    r.write(OUT / "results-B.md")
