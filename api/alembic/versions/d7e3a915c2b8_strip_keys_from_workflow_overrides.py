"""VOZ-G0-11: strip organization API keys copied into legacy workflow model overrides

`enrich_overrides_with_api_keys` stamped the org's provider keys into the legacy
`model_overrides` sections on every save. A copy is deleted only where the runtime gets the
same key back from the organization: the section runs the organization's own provider (the
runtime merges it onto the org's section, resolve.py) and holds exactly the organization's key
(MODEL_CONFIGURATION_V2, read here with SQL). A cross-provider section or a key of the
workflow's own is not a copy and stays.

Not touched (review r1 F1):
- `model_configuration_v2_override`: the runtime compiles it on its own, without the
  organization (ai_model_configuration.get_effective_ai_model_configuration_for_workflow), and
  the PUT validates the stored one; its keys go with the credential object (VOZ-AC-B1-77).
- `workflow_runs.effective_configurations`: a finished run is read again after the call
  (post-call QA builds its LLM from the frozen document, tasks/run_integrations.py), so frozen
  documents stay as they ran.

After each table's pass the candidates are read again from the database and the migration
aborts (rolling back, Postgres DDL/DML is transactional) if any still holds a copy.

Revision ID: d7e3a915c2b8
Revises: 5be1d27c9a43
Create Date: 2026-10-05 10:00:00.000000

"""

import json
import logging
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d7e3a915c2b8"
down_revision: Union[str, None] = "5be1d27c9a43"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

BATCH_SIZE = 500

# Frozen copies, as of this revision: migrations do not import application code.
_SECRET_LEAF_NAMES: tuple[str, ...] = (  # secrets_registry.SECRET_LEAF_NAMES
    "api_key",
    "credentials",
    "aws_access_key",
    "aws_secret_key",
    "aws_session_token",
)
_MODEL_OVERRIDE_SECTIONS: tuple[str, ...] = ("llm", "tts", "stt", "realtime")
_ORG_MODEL_CONFIGURATION_KEY = "MODEL_CONFIGURATION_V2"

# (table, SELECT of id, owning organization id and the document). The columns are `json`, not
# `jsonb`: `->>` renders a JSON null as SQL NULL (`->` does not). A document stored
# double-encoded is a JSON string: a candidate too, decoded below.
_TARGETS: tuple[tuple[str, str], ...] = (
    (
        "workflows",
        (
            "SELECT t.id, t.organization_id, t.workflow_configurations, "
            "json_typeof(t.workflow_configurations) = 'string' FROM workflows t"
        ),
    ),
    (
        "workflow_definitions",
        "SELECT t.id, w.organization_id, t.workflow_configurations, "
        "json_typeof(t.workflow_configurations) = 'string' "
        "FROM workflow_definitions t LEFT JOIN workflows w ON w.id = t.workflow_id",
    ),
)
_CANDIDATE = (
    "t.workflow_configurations->>'model_overrides' IS NOT NULL "
    "OR json_typeof(t.workflow_configurations) = 'string'"
)


def _load(raw):
    """The driver may hand back text; a document stored double-encoded needs a second decode."""
    for _ in range(2):
        if not isinstance(raw, str):
            break
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


# Sections each BYOK mode requires (schemas.ai_model_configuration BYOKPipeline/BYOKRealtime),
# and which of them must not run the dograh provider (_reject_dograh_provider).
_BYOK_REQUIRED: dict[str, tuple[str, ...]] = {
    "pipeline": ("llm", "tts", "stt"),
    "realtime": ("realtime", "llm"),
}
_BYOK_NO_DOGRAH = ("llm", "tts", "stt", "embeddings")


def _org_sections(document: Any) -> dict[str, dict[str, Any]]:
    """Per-section provider and keys of an org MODEL_CONFIGURATION_V2 value; a frozen copy of
    compile_ai_model_configuration_v2. ``{}`` (nothing is stripped) unless the value is one the
    runtime can load: it reads the column as the driver decodes it, with no second decode, and
    falls back to no org configuration when OrganizationAIModelConfigurationV2.model_validate
    fails (ai_model_configuration._parse_organization_ai_model_configuration_v2). The checks
    below are that schema's structure; per-provider field validation is not replicated."""
    if not isinstance(document, dict) or document.get("version", 2) != 2:
        return {}
    if document.get("mode") == "dograh":
        api_key = (document.get("dograh") or {}).get("api_key")
        if not isinstance(api_key, str):
            return {}
        return {
            s: {"provider": "dograh", "api_key": api_key} for s in ("llm", "tts", "stt")
        }
    byok = document.get("byok") if document.get("mode") == "byok" else None
    if not isinstance(byok, dict) or byok.get("mode") not in _BYOK_REQUIRED:
        return {}
    branch = byok.get(byok["mode"])
    if not isinstance(branch, dict):
        return {}
    for name in (*_BYOK_REQUIRED[byok["mode"]], "embeddings"):
        section = branch.get(name)
        if section is None and name == "embeddings":  # optional
            continue
        if not isinstance(section, dict) or not isinstance(
            section.get("provider"), str
        ):
            return {}
        if name in _BYOK_NO_DOGRAH and section["provider"] == "dograh":
            return {}
    return {
        name: section
        for name, section in branch.items()
        if name in _MODEL_OVERRIDE_SECTIONS and isinstance(section, dict)
    }


def _true_copies(doc: dict, org: dict[str, dict[str, Any]]) -> list[tuple[dict, str]]:
    """``(section, leaf)`` of every legacy override secret the organization holds verbatim."""
    overrides = doc.get("model_overrides")
    if not isinstance(overrides, dict):
        return []
    copies = []
    for name in _MODEL_OVERRIDE_SECTIONS:
        section, org_section = overrides.get(name), org.get(name)
        if not isinstance(section, dict) or not org_section:
            continue
        # Without a provider the runtime merges the section onto the org's as well.
        if section.get("provider", org_section.get("provider")) != org_section.get(
            "provider"
        ):
            continue
        copies.extend(
            (section, leaf)
            for leaf in _SECRET_LEAF_NAMES
            if section.get(leaf) is not None
            and section.get(leaf) == org_section.get(leaf)
        )
    return copies


class _OrgSections:
    def __init__(self, conn):
        self._conn, self._cache = conn, {}

    def __call__(self, organization_id: int | None) -> dict[str, dict[str, Any]]:
        if organization_id is None:
            return {}
        if organization_id not in self._cache:
            raw = self._conn.execute(
                sa.text(
                    "SELECT value FROM organization_configurations "
                    "WHERE organization_id = :org AND key = :key"
                ),
                {"org": organization_id, "key": _ORG_MODEL_CONFIGURATION_KEY},
            ).scalar_one_or_none()
            self._cache[organization_id] = _org_sections(raw)
        return self._cache[organization_id]


def _candidates(conn, table: str, select: str):
    """Keyset-paged ``(id, organization id, document, stored as a JSON string)``; a row that is
    not a JSON object is skipped and logged."""
    last_id = 0
    while True:
        rows = conn.execute(
            sa.text(
                f"{select} WHERE t.id > :last_id "
                f"AND t.workflow_configurations IS NOT NULL AND ({_CANDIDATE}) "
                f"ORDER BY t.id LIMIT :limit"
            ),
            {"last_id": last_id, "limit": BATCH_SIZE},
        ).fetchall()
        if not rows:
            return
        for row_id, organization_id, raw, double_encoded in rows:
            last_id = row_id
            doc = _load(raw)
            if doc is None:
                logger.warning(
                    "VOZ-G0-11 %s id=%d: not a JSON object, skipped", table, row_id
                )
                continue
            yield row_id, organization_id, doc, double_encoded


def _write(conn, table: str, row_id: int, doc: dict, double_encoded: bool) -> None:
    # The runtime reads a JSON-string document as no document at all; writing it back decoded
    # would switch its settings on, so it keeps its encoding.
    stored = json.dumps(json.dumps(doc)) if double_encoded else json.dumps(doc)
    conn.execute(
        sa.text(f"UPDATE {table} SET workflow_configurations = :doc WHERE id = :id"),
        {"doc": stored, "id": row_id},
    )


def _strip_table(conn, table: str, select: str, org_sections) -> int:
    total = 0
    for row_id, organization_id, doc, double_encoded in _candidates(
        conn, table, select
    ):
        copies = _true_copies(doc, org_sections(organization_id))
        if not copies:
            continue
        for section, leaf in copies:
            section.pop(leaf, None)
        _write(conn, table, row_id, doc, double_encoded)
        logger.info(
            "VOZ-G0-11 %s id=%d: copied keys removed=%d", table, row_id, len(copies)
        )
        total += len(copies)
    return total


def upgrade() -> None:
    conn = op.get_bind()
    org_sections = _OrgSections(conn)
    for table, select in _TARGETS:
        total = _strip_table(conn, table, select, org_sections)
        left = sum(
            len(_true_copies(doc, org_sections(organization_id)))
            for _, organization_id, doc, _ in _candidates(conn, table, select)
        )
        logger.info(
            "VOZ-G0-11 %s: copied keys removed=%d still stored=%d", table, total, left
        )
        if left:
            raise RuntimeError(
                f"VOZ-G0-11 aborted: {table} rows still hold {left} copied org keys"
            )


def downgrade() -> None:
    # One-way and deliberate: the removed keys are verbatim copies of the organization's own and
    # are not recoverable from the documents. Rollback is the pg_dump taken before this
    # migration (VOZ-G0-02).
    pass
