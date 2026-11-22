"""Single registry of model retirements and price changes (VOZ-AC-B1-80). Capability rows derive from it."""

from __future__ import annotations

import datetime as dt
import fnmatch
import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

_PATH = Path(__file__).with_name("deprecations.yaml")
WARN_DAYS = 30  # VOZ-AC-B1-82 default; per-entry warn_from may bring it forward
BLOCK_MARGIN = dt.timedelta(days=1)  # 422 starts 24 h before the UTC date


@dataclass(frozen=True)
class Status:
    state: str  # active | warning | warning_price | shutdown | legacy
    reason: str = ""
    hint: str = ""
    date: dt.date | None = (
        None  # retirement / price-change date; None for active and legacy
    )


@functools.cache
def load_deprecations() -> tuple[dict, ...]:
    entries = tuple(yaml.safe_load(_PATH.read_text(encoding="utf-8"))["entries"])
    for e in entries:
        _check_note(e)
    return entries


def _check_note(entry: dict) -> None:
    """The note is printed inside the one-line reason. PyYAML turns an unquoted no/yes/on/off into a bool and a block
    scalar keeps its line breaks, so anything but a non-empty one-line str is refused here instead of being dropped
    from the reason or splitting the sweep's hit."""
    note = entry.get("note")
    if note is not None and (not isinstance(note, str) or note.splitlines() != [note]):
        raise ValueError(
            f"deprecations.yaml entry {entry['model_or_feature']}: note must be a one-line string"
            " (quote it in deprecations.yaml)"
        )


def status_for(provider: str, model: str, *, today: dt.date) -> Status:
    for e in load_deprecations():
        if e["provider"] != provider or not _matches(model, _patterns(e)):
            continue
        hint = (
            f"replace with {e['replacement']}"
            if e.get("replacement")
            else "pick another model of this kind"
        )
        note = e.get(
            "note"
        )  # a scope the effect does not cover (e.g. tiers it spares), printed with the reason
        cite = f"deprecations.yaml; {note}" if note else "deprecations.yaml"
        tail = f" ({note})" if note else ""
        if e["effect"] == "legacy":
            return Status("legacy", f"{provider}/{model} is legacy ({cite})", hint)
        date = dt.date.fromisoformat(str(e["date"]))
        if e["effect"] == "price":
            return Status(
                "warning_price",
                f"{provider}/{model} price change on {date}{tail}",
                "review cost",
                date,
            )
        if today >= date - BLOCK_MARGIN:
            return Status(
                "shutdown", f"{provider}/{model} retired on {date} ({cite})", hint, date
            )
        warn_from = (
            dt.date.fromisoformat(str(e["warn_from"]))
            if e.get("warn_from")
            else date - dt.timedelta(days=WARN_DAYS)
        )
        if today >= warn_from:
            return Status(
                "warning", f"{provider}/{model} retires on {date}{tail}", hint, date
            )
    return Status("active")


MODEL_KINDS = frozenset(
    {"tts", "stt", "llm", "realtime"}
)  # entries whose names appear as model ids in saved configs


def blocking_patterns(*, today: dt.date) -> tuple[str, ...]:
    """Model-id patterns already in the shutdown window on ``today``. Legacy entries are left out: they never fail
    the sweep, so in the raw-text fallback they would only produce false ``unstructured`` hits."""
    out: list[str] = []
    for e in load_deprecations():
        if e["kind"] not in MODEL_KINDS:
            continue
        if (
            e["effect"] == "shutdown"
            and today >= dt.date.fromisoformat(str(e["date"])) - BLOCK_MARGIN
        ):
            out += _patterns(e)
    return tuple(out)


def _patterns(entry: dict) -> list[str]:
    """The entry's model_or_feature as a list: the yaml holds one pattern or a list of them."""
    value = entry["model_or_feature"]
    return value if isinstance(value, list) else [value]


def _matches(model: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(model, p) for p in patterns)
