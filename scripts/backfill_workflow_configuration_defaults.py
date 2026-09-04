"""One-off: strip schema-default leaves baked into draft workflow configurations.

Before organization-level defaults existed the UI re-sent the full
materialised document on every save, so most drafts carry every default as
if the user had set it. With the cascade live those baked values would shadow
the organization base forever. This rewrites ONLY draft definitions (never
published/archived — those are run snapshots) and only removes leaves equal
to the *schema* default, comparing Pydantic-normalised values. Surviving values
are written back normalised — trimmed strings, de-duplicated lead headers,
ints widened to floats — exactly as the workflow PUT already stores them.

A workflow with no draft is left exactly as it is: its published definition is
pinned by runs, and the legacy ``workflows.workflow_configurations`` column is
re-synced from the draft by ``WorkflowClient.save_workflow_draft`` on the next
save.

    python -m scripts.backfill_workflow_configuration_defaults            # dry-run report
    python -m scripts.backfill_workflow_configuration_defaults --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from pydantic import ValidationError

from api.db import db_client
from api.schemas.workflow_configurations import (
    WorkflowConfigurationDefaults,
    schema_defaults_document,
)
from api.services.configuration.cascade import PASSTHROUGH_KEYS, normalize_root_nulls

BATCH = 200
_ABSENT = object()


def _strip(value: Any, default: Any, path: str, removed: list[str]) -> Any:
    """Return ``value`` without the leaves equal to ``default``; ``_ABSENT``
    when nothing of the user's own is left. Keys the schema does not declare
    are kept verbatim (they carry no default to compare against)."""
    if isinstance(value, dict) and isinstance(default, dict):
        kept = {}
        for key, child in value.items():
            if key in default:
                result = _strip(child, default[key], f"{path}.{key}", removed)
                if result is not _ABSENT:
                    kept[key] = result
            else:
                kept[key] = child
        if not kept:
            removed.append(path)
            return _ABSENT
        return kept
    if value == default:
        removed.append(path)
        return _ABSENT
    return value


def strip_schema_default_leaves(
    document: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return (sparse, removed_paths). Nested null and out-of-schema keys are kept."""
    removed: list[str] = []
    stored = normalize_root_nulls(document)
    removed.extend(key for key, value in document.items() if value is None)
    normalised = WorkflowConfigurationDefaults.model_validate(stored).model_dump(
        mode="json", exclude_unset=True
    )
    defaults = schema_defaults_document()
    sparse: dict[str, Any] = {}
    for key, value in normalised.items():
        if key in PASSTHROUGH_KEYS or key not in defaults:
            sparse[key] = stored.get(key, value)
            continue
        result = _strip(value, defaults[key], key, removed)
        if result is not _ABSENT:
            sparse[key] = result
    return sparse, sorted(removed)


async def run_backfill(
    *, apply: bool, batch_size: int = BATCH
) -> dict[int, list[dict]]:
    report: dict[int, list[dict]] = {}
    after_id = 0
    while True:
        rows = await db_client.list_draft_definitions_for_backfill(
            after_id=after_id, limit=batch_size
        )
        if not rows:
            break
        for definition_id, workflow_id, organization_id, stored in rows:
            after_id = definition_id
            try:
                sparse, removed = strip_schema_default_leaves(stored)
            except ValidationError as exc:
                # Pre-existing damage (e.g. a value outside today's bounds):
                # report it, never rewrite it — the resolver clamps at run time.
                report.setdefault(organization_id, []).append(
                    {
                        "workflow_id": workflow_id,
                        "definition_id": definition_id,
                        "skipped": exc.error_count(),
                    }
                )
                continue
            if not removed:
                continue
            report.setdefault(organization_id, []).append(
                {
                    "workflow_id": workflow_id,
                    "definition_id": definition_id,
                    "removed": removed,
                    "kept": sorted(sparse),
                }
            )
            if apply:
                await db_client.update_definition_configurations(definition_id, sparse)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="write changes (default: dry-run)"
    )
    args = parser.parse_args()
    report = asyncio.run(run_backfill(apply=args.apply))
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "organizations": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
