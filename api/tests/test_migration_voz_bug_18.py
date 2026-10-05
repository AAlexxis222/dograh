"""VOZ-BUG-18 (G0 §3.2 S3): upstream's migration rewrites only workflows/definitions and writes "default",
while unknown values fall back to DEFAULT_TURN_START_STRATEGY = min_words. Migrated and unmigrated rows
must resolve to the SAME start strategy."""

from api.services.workflow.voz_bug_18 import backfill_turn_start, resolve_turn_start


def test_migrated_and_unmigrated_rows_resolve_the_same():
    migrated = backfill_turn_start({"turn_start_strategy": "provisional_vad"})
    assert resolve_turn_start(migrated) == resolve_turn_start({}) == "default"


def test_backfill_is_idempotent():
    doc = backfill_turn_start({"turn_start_strategy": "provisional_vad"})
    assert backfill_turn_start(dict(doc)) == doc


def test_backfill_preserves_explicit_min_words():
    assert backfill_turn_start({"turn_start_strategy": "min_words"})["turn_start_strategy"] == "min_words"


def test_count_before_equals_after():   # VOZ-AC-B9-38
    docs = [{"turn_start_strategy": "provisional_vad"}, {}, {"turn_start_strategy": "min_words"}]
    after = [backfill_turn_start(dict(d)) for d in docs]
    assert len(after) == len(docs) and "provisional_vad" not in str(after)
