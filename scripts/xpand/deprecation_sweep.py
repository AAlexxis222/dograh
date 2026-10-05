"""Read-only sweep: saved model configs that point at retiring or legacy models.

    DATABASE_URL=<instance> python -m scripts.xpand.deprecation_sweep --horizon-days 14

Prints one line per hit (code, where, reason, hint) and exits 1 if there is any.
Reuses api.services.capabilities.deprecations; nothing is written to the DB.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from api.services.capabilities.deprecations import status_for

CODE = "deprecated_model_in_use"
_BLOCKING = ("shutdown", "legacy")


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


def _service_refs(doc: Any, path: str = "") -> Iterator[tuple[str, str, str]]:
    """Yield (path, provider, model) for every dict carrying both strings.

    Shape-agnostic on purpose: the same {provider, model} leaf is stored under
    top-level stt/tts/llm/realtime, under model_overrides, and under the org v2
    byok.pipeline / byok.realtime sections.
    """
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
                hits.append(Hit(kind, row_id, path, provider, model, status.state, status.reason, status.hint))
    return hits


def format_hit(hit: Hit) -> str:
    return f"{CODE} where={hit.kind}#{hit.row_id}:{hit.path} reason={hit.reason} hint={hit.hint}"


_QUERIES = (
    ("workflow", "SELECT w.id, w.workflow_configurations FROM workflows w WHERE w.released_definition_id IS NOT NULL"),
    ("workflow_definition", "SELECT id, workflow_configurations FROM workflow_definitions WHERE status = 'published'"),
    ("workflow_definition", "SELECT id, workflow_json FROM workflow_definitions WHERE status = 'published'"),
    ("org", "SELECT id, value FROM organization_configurations"),
)


async def _load_docs(database_url: str) -> list[tuple[str, int, Any]]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url)
    docs: list[tuple[str, int, Any]] = []
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SET TRANSACTION READ ONLY"))
            for kind, sql in _QUERIES:
                docs += [(kind, row[0], row[1]) for row in await conn.execute(text(sql))]
    finally:
        await engine.dispose()
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon-days", type=int, default=14)
    args = parser.parse_args(argv)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("deprecation_sweep_no_database where=env:DATABASE_URL reason=variable not set hint=export DATABASE_URL (read-only access is enough)")
        return 2
    docs = asyncio.run(_load_docs(database_url))
    hits = find_deprecated_usage(docs, today=dt.date.today(), horizon_days=args.horizon_days)
    for hit in hits:
        print(format_hit(hit))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
