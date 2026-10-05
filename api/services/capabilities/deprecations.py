"""Single registry of model retirements and price changes (VOZ-AC-B1-80). Capability rows derive from it."""

from __future__ import annotations

import datetime as dt
import fnmatch
import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

_PATH = Path(__file__).with_name("deprecations.yaml")
WARN_DAYS = 30            # VOZ-AC-B1-82 default; per-entry warn_from may bring it forward
BLOCK_MARGIN = dt.timedelta(days=1)  # 422 starts 24 h before the UTC date


@dataclass(frozen=True)
class Status:
    state: str            # active | warning | warning_price | shutdown | legacy
    reason: str = ""
    hint: str = ""
    date: dt.date | None = None   # retirement / price-change date; None for active and legacy


@functools.cache
def load_deprecations() -> tuple[dict, ...]:
    return tuple(yaml.safe_load(_PATH.read_text(encoding="utf-8"))["entries"])


def status_for(provider: str, model: str, *, today: dt.date) -> Status:
    for e in load_deprecations():
        if e["provider"] != provider or not _matches(model, e["model_or_feature"]):
            continue
        hint = f"replace with {e['replacement']}" if e.get("replacement") else "pick another model of this kind"
        if e["effect"] == "legacy":
            return Status("legacy", f"{provider}/{model} is legacy (deprecations.yaml)", hint)
        date = dt.date.fromisoformat(str(e["date"]))
        if e["effect"] == "price":
            return Status("warning_price", f"{provider}/{model} price change on {date}", "review cost", date)
        if today >= date - BLOCK_MARGIN:
            return Status("shutdown", f"{provider}/{model} retired on {date} (deprecations.yaml)", hint, date)
        warn_from = dt.date.fromisoformat(str(e["warn_from"])) if e.get("warn_from") else date - dt.timedelta(days=WARN_DAYS)
        if today >= warn_from:
            return Status("warning", f"{provider}/{model} retires on {date}", hint, date)
    return Status("active")


MODEL_KINDS = frozenset({"tts", "stt", "llm", "realtime"})   # entries whose names appear as model ids in saved configs


def blocking_patterns(*, today: dt.date) -> tuple[str, ...]:
    """Model-id patterns already in the shutdown window on ``today``. Legacy entries are left out: they never fail
    the sweep, so in the raw-text fallback they would only produce false ``unstructured`` hits."""
    out: list[str] = []
    for e in load_deprecations():
        if e["kind"] not in MODEL_KINDS:
            continue
        if e["effect"] == "shutdown" and today >= dt.date.fromisoformat(str(e["date"])) - BLOCK_MARGIN:
            out += e["model_or_feature"] if isinstance(e["model_or_feature"], list) else [e["model_or_feature"]]
    return tuple(out)


def _matches(model: str, patterns) -> bool:
    patterns = patterns if isinstance(patterns, list) else [patterns]
    return any(fnmatch.fnmatchcase(model, p) for p in patterns)
