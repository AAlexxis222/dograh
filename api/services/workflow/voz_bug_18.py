"""Single mapping of legacy turn_start_strategy values (VOZ-BUG-18), shared by the alembic backfill
and by the schema reader (dual reader during the expand phase, VOZ-AC-B9-38)."""

LEGACY_TO_CURRENT = {"provisional_vad": "default"}
ORG_DEFAULT = "default"  # VOZ-AC-B2-30: org default pinned to "default", not upstream's min_words


def backfill_turn_start(doc: dict) -> dict:
    value = doc.get("turn_start_strategy")
    if value in LEGACY_TO_CURRENT:
        doc["turn_start_strategy"] = LEGACY_TO_CURRENT[value]
    return doc


def pin_org_default(doc: dict) -> dict:
    """Org defaults document: pin the strategy when it is absent or an explicit null
    (the schema treats null as unset)."""
    if doc.get("turn_start_strategy") is None:
        doc["turn_start_strategy"] = ORG_DEFAULT
    return doc


def resolve_turn_start(doc: dict) -> str:
    value = doc.get("turn_start_strategy")
    return LEGACY_TO_CURRENT.get(value, value) or ORG_DEFAULT
