import datetime as dt

from scripts.xpand.deprecation_sweep import find_deprecated_usage, format_hit


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
