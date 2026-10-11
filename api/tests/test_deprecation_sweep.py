import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from api.services.capabilities.deprecations import blocking_patterns
from scripts.xpand import deprecation_sweep as sweep
from scripts.xpand.deprecation_sweep import (
    build_text_regex,
    find_deprecated_usage,
    format_hit,
    with_unstructured,
)

TODAY = dt.date(2026, 10, 6)
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_finds_saved_model_in_published_workflow_docs():
    docs = [("workflow", 7771, {"tts": {"provider": "cartesia", "model": "sonic-2"}}),
            ("org", 3, {"tts": {"provider": "cartesia", "model": "sonic-3.6"}})]
    hits = find_deprecated_usage(docs, today=dt.date(2026, 10, 6), horizon_days=14)
    assert [(h.kind, h.row_id, h.model) for h in hits] == [("workflow", 7771, "sonic-2")]


def test_far_future_dates_are_outside_the_horizon():
    docs = [("workflow", 1, {"tts": {"provider": "openai", "model": "gpt-4o-mini-tts"}})]
    assert find_deprecated_usage(docs, today=dt.date(2026, 10, 6), horizon_days=14) == []


def test_walks_nested_v2_and_realtime_shapes_and_flags_legacy():
    docs = [
        ("org", 5, {"version": 2, "byok": {"pipeline": {"tts": {"provider": "cartesia", "model": "sonic-turbo"}}}}),
        ("workflow", 9, {"model_overrides": {"realtime": {"provider": "grok_realtime", "model": "grok-voice-think-fast-1.0"}}}),
        ("workflow_definition", 4, {"nodes": [{"data": {"tts": {"provider": "cartesia", "model": "sonic-2"}}}]}),
    ]
    hits = find_deprecated_usage(docs, today=dt.date(2026, 10, 6), horizon_days=14)
    assert [(h.row_id, h.path, h.state) for h in hits] == [
        (5, "byok.pipeline.tts", "shutdown"),
        (9, "model_overrides.realtime", "legacy"),
        (4, "nodes[0].data.tts", "shutdown"),
    ]


def test_hit_is_printed_in_b0_28_shape():
    (hit,) = find_deprecated_usage(
        [("workflow", 7771, {"tts": {"provider": "cartesia", "model": "sonic-2"}})],
        today=dt.date(2026, 10, 6), horizon_days=14,
    )
    line = format_hit(hit)
    assert line.startswith("deprecated_model_in_use ")
    for field in ("where=workflow#7771:tts", "reason=", "hint=replace with sonic-3.6"):
        assert field in line


def _models(docs):
    return sorted(h.model for h in find_deprecated_usage(docs, today=TODAY, horizon_days=14))


def test_dated_cartesia_snapshots_are_found():
    docs = [
        ("workflow", 1, {"tts": {"provider": "cartesia", "model": "sonic-2-2025-06-11"}}),
        ("workflow", 2, {"tts": {"provider": "cartesia", "model": "sonic-turbo-2025-06-04"}}),
        ("workflow", 3, {"tts": {"provider": "cartesia", "model": "sonic-3-2026-01-12"}}),
        ("workflow", 4, {"tts": {"provider": "cartesia", "model": "sonic-3.6-2026-08-27"}}),
    ]
    assert _models(docs) == ["sonic-2-2025-06-11", "sonic-turbo-2025-06-04"]


def test_already_sunset_cartesia_models_are_found():
    docs = [("org", i, {"tts": {"provider": "cartesia", "model": m}})
            for i, m in enumerate(["sonic", "sonic-english", "sonic-multilingual", "sonic-2024-12-12"])]
    assert len(_models(docs)) == 4


def test_json_string_payloads_are_decoded_even_when_double_encoded():
    inner = {"tts": {"provider": "cartesia", "model": "sonic-2"}}
    docs = [("workflow", 1, json.dumps(inner)),
            ("workflow", 2, {"model_overrides": json.dumps(inner)}),
            ("workflow", 3, json.dumps(json.dumps(inner))),
            ("workflow", 4, {"note": "{not json", "other": "[1, 2]"})]
    hits = find_deprecated_usage(docs, today=TODAY, horizon_days=14)
    assert [(h.row_id, h.path) for h in hits] == [(1, "tts"), (2, "model_overrides.tts"), (3, "tts")]


def test_text_regex_matches_quoted_model_values_only():
    rx = build_text_regex(["sonic-2", "sonic-2-*", "o3*"])
    assert re.search(rx, json.dumps({"tts": {"model": "sonic-2"}}), re.I)
    assert re.search(rx, json.dumps({"model": "sonic-2-2025-06-11"}), re.I)
    assert re.search(rx, json.dumps(json.dumps({"model": "SONIC-2"})), re.I)  # double-encoded, any case
    assert not re.search(rx, json.dumps({"model": "sonic-20"}), re.I)
    assert not re.search(rx, json.dumps({"prompt": "say sonic-2 and sonic-2-x aloud"}), re.I)
    assert re.search(rx, json.dumps({"model": "o3-mini"}), re.I)
    span = build_text_regex(["gpt-5*-2025-08-07"])
    assert re.search(span, json.dumps({"model": "gpt-5-mini-2025-08-07"}), re.I)
    assert not re.search(span, json.dumps({"a": "gpt-5-mini", "b": "x-2025-08-07"}), re.I)  # a glob never spans two strings


def test_text_regex_rejects_unsupported_glob_syntax():
    with pytest.raises(ValueError):
        build_text_regex(["sonic-?"])


def test_blocking_patterns_follow_horizon_and_model_kinds():
    now = blocking_patterns(today=TODAY + dt.timedelta(days=14))
    assert {"sonic-2", "sonic-2-*", "sonic-turbo-*", "sonic-3-2025-10-27"} <= set(now)
    assert "grok-voice-think-fast-1.0" not in now      # legacy never fails the sweep, so no text fallback
    assert not {"o3*", "line-sdk-hosting", "v1/prompts", "eleven_v4"} & set(now)   # not yet in their window
    later = set(blocking_patterns(today=dt.date(2027, 6, 1)))
    assert "o3*" in later
    assert not {"line-sdk-hosting", "v1/prompts", "evals", "voice-sdk", "eleven_v4", "gemini-tts-promo-pricing"} & later


def test_row_missed_by_structured_walk_becomes_unstructured_hit():
    structured = find_deprecated_usage(
        [("workflow", 1, {"tts": {"provider": "cartesia", "model": "sonic-2"}}),
         ("workflow", 2, {"model_overrides": {"tts": {"model": "sonic-2"}}})],
        today=TODAY, horizon_days=14)
    assert [h.row_id for h in structured] == [1]
    merged = with_unstructured(structured, [("workflow", 1), ("workflow", 2), ("org", 2)])
    assert [(h.kind, h.row_id, h.state) for h in merged] == [
        ("workflow", 1, "shutdown"), ("org", 2, "unstructured"), ("workflow", 2, "unstructured")]
    line = format_hit(merged[2])
    assert line.startswith("unstructured_hit where=workflow#2:") and "state=unstructured" in line
    assert "reason=" in line and "hint=" in line


def test_legacy_structured_hit_does_not_hide_provider_less_shutdown_in_same_row():
    doc = {"model_overrides": {"stt": {"provider": "assemblyai", "model": "u3-rt-pro"}, "tts": {"model": "sonic-2"}}}
    structured = find_deprecated_usage([("workflow_definition", 107, doc)], today=TODAY, horizon_days=14)
    assert [h.state for h in structured] == ["legacy"]
    merged = with_unstructured(structured, [("workflow_definition", 107)])
    assert [h.state for h in merged] == ["legacy", "unstructured"]


def test_lines_carry_state_and_date():
    docs = [("workflow", 1, {"tts": {"provider": "cartesia", "model": "sonic-2"}}),
            ("workflow", 2, {"realtime": {"provider": "grok_realtime", "model": "grok-voice-think-fast-1.0"}})]
    a, b = find_deprecated_usage(docs, today=TODAY, horizon_days=14)
    assert "state=shutdown date=2026-10-20" in format_hit(a)
    assert "state=legacy date=none" in format_hit(b)


def _run_main(monkeypatch, capsys, docs, text_rows=()):
    async def fake_load(url, pattern):
        return list(docs), list(text_rows)

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://unused")
    monkeypatch.setattr(sweep, "_load_rows", fake_load)
    code = sweep.main(["--horizon-days", "3650"])
    return code, capsys.readouterr().out


def test_exit_code_is_1_for_shutdown(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, [("workflow", 1, {"tts": {"provider": "cartesia", "model": "sonic-2"}})])
    assert code == 1 and "state=shutdown" in out


def test_exit_code_is_0_when_only_legacy(monkeypatch, capsys):
    docs = [("workflow", 1, {"realtime": {"provider": "grok_realtime", "model": "grok-voice-think-fast-1.0"}})]
    code, out = _run_main(monkeypatch, capsys, docs)
    assert code == 0 and "state=legacy" in out


def test_legacy_only_row_exits_0_with_the_real_text_regex(monkeypatch, capsys):
    rx = build_text_regex(blocking_patterns(today=TODAY + dt.timedelta(days=14)))
    docs = [("workflow", 1, {"stt": {"provider": "assemblyai", "model": "u3-rt-pro"}}),
            ("workflow", 2, {"realtime": {"provider": "grok_realtime", "model": "grok-voice-think-fast-1.0"}})]
    text_rows = [(k, i) for k, i, doc in docs if re.search(rx, json.dumps(doc), re.I)]
    code, out = _run_main(monkeypatch, capsys, docs, text_rows=text_rows)
    assert code == 0 and "unstructured_hit" not in out


def test_exit_code_is_1_for_unstructured_only(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, [("workflow", 9, {"x": 1})], text_rows=[("workflow", 9)])
    assert code == 1 and "unstructured_hit" in out


def test_exit_code_is_0_with_no_hits(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, [("workflow", 1, {"tts": {"provider": "cartesia", "model": "sonic-3.6"}})])
    assert code == 0 and out == ""


def test_utc_today_uses_utc_not_local_time():
    def clock(tz=None):  # 23:30 at UTC-5 is already the next day in UTC
        base = dt.datetime(2026, 10, 6, 23, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
        return base.astimezone(tz) if tz else base.replace(tzinfo=None)

    assert sweep._utc_today(clock) == dt.date(2026, 10, 7)


def test_sources_cover_draft_definitions_templates_and_user_configs_with_real_columns():
    from api.db.models import Base

    kinds = {s.kind for s in sweep._SOURCES}
    assert {"workflow_definition", "workflow_definition_json", "workflow_template", "user_config", "org", "workflow"} <= kinds
    for s in sweep._SOURCES:
        assert s.column in Base.metadata.tables[s.table].c, s
        if s.table == "workflow_definitions":
            assert "'draft'" in s.where and "'published'" in s.where


def test_script_runs_from_repo_root_without_pythonpath():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "DATABASE_URL")}
    r = subprocess.run([sys.executable, "scripts/xpand/deprecation_sweep.py"], cwd=REPO_ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "deprecation_sweep_no_database" in r.stdout, r.stderr


# --- documents encrypted at rest (VOZ-N0-24c) ---------------------------------------------------------------------


def _key(monkeypatch, value):
    from api.services.security import credential_box

    if value is None:
        monkeypatch.delenv(credential_box.ENV, raising=False)
    else:
        monkeypatch.setenv(credential_box.ENV, value)
    credential_box.load_keys.cache_clear()


def _whole_token(monkeypatch, table, row_id, column, document):
    """A document stored whole as one token, as the encryption at rest writes it."""
    import base64

    from api.services.security import credential_box

    key = "k1=" + base64.b64encode(os.urandom(32)).decode()
    _key(monkeypatch, key)
    plaintext = json.dumps(document, separators=(",", ":")).encode()
    token = credential_box.encrypt(plaintext, credential_box.aad_for(table, row_id, column, ()))
    return key, token


def test_an_encrypted_org_configuration_is_opened_and_swept(monkeypatch, capsys):
    document = {"byok": {"pipeline": {"tts": {"provider": "cartesia", "model": "sonic-2", "api_key": "x" * 20}}}}
    _key_value, token = _whole_token(monkeypatch, "organization_configurations", 5, "value", document)
    code, out = _run_main(monkeypatch, capsys, [("org", 5, token)])
    assert code == 1
    assert "where=org#5:byok.pipeline.tts state=shutdown" in out


def test_an_encrypted_user_configuration_goes_through_the_text_regex(monkeypatch, capsys):
    _key_value, token = _whole_token(
        monkeypatch, "user_configurations", 8, "configuration", {"model_overrides": {"tts": "sonic-2"}}
    )
    code, out = _run_main(monkeypatch, capsys, [("user_config", 8, token)])
    assert code == 1 and "unstructured_hit where=user_config#8" in out


@pytest.mark.parametrize("key", [None, "wrong"])
def test_a_token_the_sweep_cannot_open_exits_non_zero(monkeypatch, capsys, key):
    import base64

    _key_value, token = _whole_token(
        monkeypatch, "organization_configurations", 5, "value", {"tts": {"provider": "cartesia", "model": "sonic-3.6"}}
    )
    _key(monkeypatch, None if key is None else "k1=" + base64.b64encode(os.urandom(32)).decode())
    code, out = _run_main(monkeypatch, capsys, [("org", 5, token), ("workflow", 1, {"x": 1})])
    assert code == 3
    (line,) = [line for line in out.splitlines() if line.startswith("deprecation_sweep_encrypted_unreadable")]
    assert "where=org#5" in line and "reason=" in line and "hint=" in line
    assert token not in out


def test_a_double_encoded_token_text_is_opened_too(monkeypatch, capsys):
    document = {"tts": {"provider": "cartesia", "model": "sonic-2"}}
    _key_value, token = _whole_token(monkeypatch, "organization_configurations", 6, "value", document)
    code, out = _run_main(monkeypatch, capsys, [("org", 6, json.dumps(token))])
    assert code == 1 and "where=org#6:tts state=shutdown" in out


def _leaf_sealed_workflow(monkeypatch):
    """A workflow configuration whose secret leaf (a ``credentials`` object) is a token, as the app stores it."""
    import base64

    from api.services.security.credential_documents import seal_document

    _key(monkeypatch, "k1=" + base64.b64encode(os.urandom(32)).decode())
    document = {"model_configuration_v2_override": {"tts": {"credentials": {"provider": "cartesia", "model": "sonic-2"}}}}
    sealed = seal_document("workflows", 4, "workflow_configurations", document)
    assert sealed["model_configuration_v2_override"]["tts"]["credentials"].startswith("v1:")
    return sealed


def test_a_leaf_token_inside_a_document_is_opened(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, [("workflow", 4, _leaf_sealed_workflow(monkeypatch))])
    assert code == 1
    assert "where=workflow#4:model_configuration_v2_override.tts.credentials state=shutdown" in out


def test_a_leaf_token_the_sweep_cannot_open_exits_non_zero(monkeypatch, capsys):
    sealed = _leaf_sealed_workflow(monkeypatch)
    _key(monkeypatch, None)
    code, out = _run_main(monkeypatch, capsys, [("workflow", 4, sealed)])
    assert code == 3
    (line,) = [line for line in out.splitlines() if line.startswith("deprecation_sweep_encrypted_unreadable")]
    assert "where=workflow#4" in line and "reason=" in line and "hint=" in line
