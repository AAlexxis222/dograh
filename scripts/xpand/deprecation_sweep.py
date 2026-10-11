"""Read-only sweep: saved model configs that point at retiring or legacy models.

From the repo root (both forms work; the DB URL must use an async driver, e.g. postgresql+asyncpg://):

    DATABASE_URL=<instance> python scripts/xpand/deprecation_sweep.py --horizon-days 14
    DATABASE_URL=<instance> python -m scripts.xpand.deprecation_sweep --horizon-days 14

One line per hit: <code> where=<kind>#<id>:<path> state=<state> date=<date|none> reason=... hint=...
Exit 1 if there is a shutdown (or unstructured) hit within the horizon, 0 if there are none or only legacy ones,
2 if DATABASE_URL is missing, 3 if a row is encrypted at rest and cannot be opened (no or wrong
CREDENTIALS_MASTER_KEY): the sweep never reports a clean result for a document it could not read. The horizon must reach the shutdown date: run it with
--horizon-days >= days until the date (e.g. 14 on or after 2026-10-06 for the 2026-10-20 Cartesia shutdown);
a shorter horizon reports nothing for that model. Reuses api.services.capabilities.deprecations; nothing is written to the DB.

Two passes per table, erring on the side of reporting: a structured walk over {provider, model} pairs (JSON
strings are decoded), and a raw-text regex built from the registry that catches what the walk cannot
interpret (e.g. a provider-less model_overrides). Rows only the regex finds print as ``unstructured_hit``.

Encryption at rest (VOZ-N0-24c) stores some documents whole as one credential_box token (the org model
configuration, the user model configuration): those are opened with CREDENTIALS_MASTER_KEY at the whole-column AAD
(``table:id:column:[]``) and run through both passes in Python. Secret leaves inside a document carry no model ids.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __name__ == "__main__":  # `python scripts/xpand/deprecation_sweep.py` puts scripts/xpand, not the repo root, on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.services.capabilities.deprecations import blocking_patterns, status_for  # noqa: E402
from api.services.security import credential_box

CODE = "deprecated_model_in_use"
UNREADABLE_CODE = "deprecation_sweep_encrypted_unreadable"
_TOKEN = re.compile(credential_box.TOKEN_PATTERN)
UNSTRUCTURED_CODE = "unstructured_hit"
_BLOCKING = ("shutdown", "legacy")
_FAILING = ("shutdown", "unstructured")   # exit 1; legacy alone only prints


@dataclass(frozen=True)
class Hit:
    kind: str
    row_id: int
    path: str
    provider: str
    model: str
    state: str
    reason: str
    hint: str
    date: dt.date | None = None


def _decode(value: Any) -> Any:
    """JSON objects/arrays stored as strings (possibly several layers deep) become Python values."""
    for _ in range(4):  # bounded: each pass peels one layer of encoding
        if not (isinstance(value, str) and value.lstrip()[:1] in ('{', '[', '"')):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value


def _service_refs(doc: Any, path: str = "") -> Iterator[tuple[str, str, str]]:
    """Yield (path, provider, model) for every dict carrying both strings.

    Shape-agnostic on purpose: the same {provider, model} leaf is stored under
    top-level stt/tts/llm/realtime, under model_overrides, and under the org v2
    byok.pipeline / byok.realtime sections.
    """
    doc = _decode(doc)
    if isinstance(doc, dict):
        provider, model = doc.get("provider"), doc.get("model")
        if isinstance(provider, str) and isinstance(model, str):
            yield path, provider, model
            return
        for key, child in doc.items():
            yield from _service_refs(child, f"{path}.{key}" if path else str(key))
    elif isinstance(doc, list):
        for i, child in enumerate(doc):
            yield from _service_refs(child, f"{path}[{i}]")


def find_deprecated_usage(
    docs: Iterable[tuple[str, int, Any]], *, today: dt.date, horizon_days: int
) -> list[Hit]:
    """Models that are shutdown/legacy once ``horizon_days`` have passed."""
    at = today + dt.timedelta(days=horizon_days)
    hits = []
    for kind, row_id, doc in docs:
        for path, provider, model in _service_refs(doc):
            status = status_for(provider, model, today=at)
            if status.state in _BLOCKING:
                hits.append(Hit(kind, row_id, path, provider, model, status.state, status.reason, status.hint, status.date))
    return hits


def build_text_regex(patterns: Iterable[str]) -> str:
    """Regex (Python re and Postgres ARE, use case-insensitively) matching a quoted JSON string equal to a pattern.

    ``\\*"`` tolerates the backslashes of JSON nested inside JSON strings; ``*`` in a pattern spans any non-quote run.
    """
    alternatives = []
    for pattern in patterns:
        if "?" in pattern or "[" in pattern:
            raise ValueError(f"unsupported glob syntax in deprecations.yaml pattern: {pattern!r}")
        alternatives.append('[^"\\\\]*'.join(re.escape(part) for part in pattern.split("*")))
    if not alternatives:
        raise ValueError("no patterns to build a text regex from")
    return '\\\\*"(' + "|".join(alternatives) + ')\\\\*"'


def with_unstructured(hits: list[Hit], text_rows: Iterable[tuple[str, int]]) -> list[Hit]:
    """Add an ``unstructured`` hit for each (kind, row_id) the text regex found but the structured walk did not
    report as shutdown (a legacy-only structured hit must not hide a provider-less retiring model in the same row)."""
    seen = {(h.kind, h.row_id) for h in hits if h.state == "shutdown"}
    extra = [
        Hit(kind, row_id, "<raw-json-text>", "?", "?", "unstructured",
            "a retiring or legacy model id appears in the row but not inside a {provider, model} pair",
            "open the full row; likely a provider-less model_overrides or a non-standard shape")
        for kind, row_id in sorted(set(text_rows) - seen)
    ]
    return hits + extra


def format_hit(hit: Hit) -> str:
    code = UNSTRUCTURED_CODE if hit.state == "unstructured" else CODE
    date = hit.date.isoformat() if hit.date else "none"
    return f"{code} where={hit.kind}#{hit.row_id}:{hit.path} state={hit.state} date={date} reason={hit.reason} hint={hit.hint}"


@dataclass(frozen=True)
class Source:
    kind: str
    table: str
    column: str
    where: str = ""


_SOURCES = (
    Source("workflow", "workflows", "workflow_configurations", "WHERE released_definition_id IS NOT NULL"),
    Source("workflow_definition", "workflow_definitions", "workflow_configurations", "WHERE status IN ('draft', 'published')"),
    Source("workflow_definition_json", "workflow_definitions", "workflow_json", "WHERE status IN ('draft', 'published')"),
    Source("workflow_template", "workflow_templates", "template_json"),
    Source("org", "organization_configurations", "value"),
    Source("user_config", "user_configurations", "configuration"),
)


async def _load_rows(database_url: str, pattern: str) -> tuple[list[tuple[str, int, Any]], list[tuple[str, int]]]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url)
    docs: list[tuple[str, int, Any]] = []
    text_rows: list[tuple[str, int]] = []
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SET TRANSACTION READ ONLY"))
            for s in _SOURCES:  # identifiers come from the constant above, the pattern is bound
                sql = text(f"SELECT id, {s.column}, CAST({s.column} AS text) ~* :pat FROM {s.table} {s.where}")
                for row_id, doc, text_hit in await conn.execute(sql, {"pat": pattern}):
                    docs.append((s.kind, row_id, doc))
                    if text_hit:
                        text_rows.append((s.kind, row_id))
    finally:
        await engine.dispose()
    return docs, text_rows


def open_encrypted_rows(
    docs: list[tuple[str, int, Any]], text_rows: list[tuple[str, int]], pattern: str
) -> tuple[list[tuple[str, int, Any]], list[tuple[str, int]], list[str]]:
    """Open every document stored whole as a token; the text regex runs on the opened JSON. Returns the docs, the
    text rows and one B0-28 line per document that could not be opened."""
    tables = {s.kind: s for s in _SOURCES}
    opened, rows, unreadable = [], list(text_rows), []
    for kind, row_id, doc in docs:
        token = _decode(doc)
        if not (isinstance(token, str) and _TOKEN.fullmatch(token)):
            opened.append((kind, row_id, doc))
            continue
        source = tables[kind]
        aad = credential_box.aad_for(source.table, row_id, source.column, ())
        try:
            plain = json.loads(credential_box.decrypt(token, aad))
        except (credential_box.CredentialUnreadable, credential_box.MasterKeyInvalid) as e:
            unreadable.append(
                f"{UNREADABLE_CODE} where={kind}#{row_id} reason=the document is encrypted at rest and "
                f"cannot be opened ({e.reason}) hint=run the sweep with the CREDENTIALS_MASTER_KEY of this database"
            )
            continue
        opened.append((kind, row_id, plain))
        if re.search(pattern, json.dumps(plain), re.IGNORECASE):
            rows.append((kind, row_id))
    return opened, rows, unreadable


def _utc_today(clock: Callable[..., dt.datetime] = dt.datetime.now) -> dt.date:
    return clock(dt.timezone.utc).date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon-days", type=int, default=14)
    args = parser.parse_args(argv)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("deprecation_sweep_no_database where=env:DATABASE_URL reason=variable not set hint=export DATABASE_URL (read-only access is enough)")
        return 2
    today = _utc_today()
    pattern = build_text_regex(blocking_patterns(today=today + dt.timedelta(days=args.horizon_days)))
    docs, text_rows = asyncio.run(_load_rows(database_url, pattern))
    docs, text_rows, unreadable = open_encrypted_rows(docs, text_rows, pattern)
    hits = with_unstructured(find_deprecated_usage(docs, today=today, horizon_days=args.horizon_days), text_rows)
    for hit in hits:
        print(format_hit(hit))
    for line in unreadable:
        print(line)
    if unreadable:
        return 3
    return 1 if any(h.state in _FAILING for h in hits) else 0


if __name__ == "__main__":
    sys.exit(main())
