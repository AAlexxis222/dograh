"""Session-scoped results table for the scenario runs (spec §4)."""
from pathlib import Path

import pytest

OUT = Path(__file__).parent / "out"
COLUMNS = ["scenario", "mode", "wait_ms", "offset", "messages", "texts", "down_started",
           "down_stopped", "interruptions", "lost", "ghost", "stats"]


class Results:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, **row):
        self.rows.append(row)


@pytest.fixture(scope="session")
def results():
    r = Results()
    yield r
    OUT.mkdir(exist_ok=True)
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in r.rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in COLUMNS) + " |")
    (OUT / "results-A.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
