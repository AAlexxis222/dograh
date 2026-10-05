"""VOZ-AC-B1-81: no registry default points to a model with a past date or legacy/deprecated status."""

import datetime as dt
import inspect

import pytest
from pydantic import BaseModel

from api.services.capabilities.deprecations import load_deprecations, status_for
from api.services.configuration import registry

TODAY = dt.date(2026, 10, 6)

# Defaults that still point to dead models; VOZ-N0-02 fixes them and removes this set.
_FIXED_BY_N0_02 = {
    "AssemblyAISTTConfiguration",
    # GoogleRealtimeLLMConfiguration: upstream #833 (merged in G0) moved its default to gemini-3.8-live.
    "GoogleVertexRealtimeLLMConfiguration",
    "GrokRealtimeLLMConfiguration",
    "OpenAISTTConfiguration",
    "OpenAITTSService",
}


def _configs():
    for name, cls in inspect.getmembers(registry, inspect.isclass):
        if issubclass(cls, BaseModel) and cls.__module__ == registry.__name__ and "model" in cls.model_fields \
                and "provider" in cls.model_fields:
            yield name, cls


@pytest.mark.parametrize(
    "name,cls",
    [
        pytest.param(n, c, marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason="fixed by VOZ-N0-02"))
        if n in _FIXED_BY_N0_02
        else (n, c)
        for n, c in _configs()
    ],
)
def test_registry_defaults_not_deprecated(name, cls):
    provider = str(getattr(cls.model_fields["provider"].default, "value", cls.model_fields["provider"].default))
    model = cls.model_fields["model"].default
    status = status_for(provider, model, today=TODAY)
    assert status.state in {"active", "warning_price"}, (
        f"{name}: default model {provider}/{model} is {status.state} ({status.reason}); hint: {status.hint}"
    )


def test_price_effect_only_warns():
    s = status_for("elevenlabs", "eleven_v4_turbo", today=dt.date(2026, 10, 13))
    assert s.state == "warning_price"


def test_shutdown_is_blocking_from_24h_before_date():
    assert status_for("cartesia", "sonic-2", today=dt.date(2026, 10, 19)).state == "shutdown"


def test_every_entry_has_source_and_read_at():
    for e in load_deprecations():
        assert e["source_url"].startswith("http") and e["read_at"], e


def test_legacy_entry_has_no_date_and_is_legacy():
    s = status_for("grok_realtime", "grok-voice-think-fast-1.0", today=TODAY)
    assert s.state == "legacy" and "grok-voice-latest" in s.hint


@pytest.mark.parametrize("model", [
    "sonic-2", "sonic-2-2025-06-11", "sonic-turbo", "sonic-turbo-2025-06-04", "sonic-3-2025-10-27",
])
def test_cartesia_sunset_models_and_dated_snapshots_are_shutdown(model):
    assert status_for("cartesia", model, today=dt.date(2026, 10, 20)).state == "shutdown"


@pytest.mark.parametrize("model", ["sonic", "sonic-english", "sonic-multilingual", "sonic-2024-12-12", "sonic-2024-10-19"])
def test_cartesia_models_already_sunset_in_june_are_shutdown_now(model):
    assert status_for("cartesia", model, today=TODAY).state == "shutdown"


@pytest.mark.parametrize("model", ["sonic-3", "sonic-3-2026-01-12", "sonic-3.5", "sonic-3.6", "sonic-3.6-2026-08-27", "sonic-preview"])
def test_supported_cartesia_models_stay_active(model):
    assert status_for("cartesia", model, today=dt.date(2026, 10, 25)).state == "active"


@pytest.mark.parametrize("provider", ["openai_realtime", "azure_realtime"])
@pytest.mark.parametrize("model", [
    "gpt-realtime", "gpt-realtime-2025-08-28", "gpt-realtime-mini", "gpt-realtime-mini-2025-12-15", "gpt-4o-realtime-preview",
])
def test_realtime_retirement_uses_registry_provider_keys(provider, model):
    assert status_for(provider, model, today=dt.date(2027, 1, 20)).state == "shutdown"


@pytest.mark.parametrize("model", ["gpt-realtime-2", "gpt-realtime-2.1", "gpt-realtime-2.1-mini"])
def test_realtime_replacements_are_not_flagged(model):
    assert status_for("openai_realtime", model, today=dt.date(2027, 2, 1)).state == "active"


def test_openai_provider_key_no_longer_carries_realtime_entry():
    assert status_for("openai", "gpt-realtime", today=dt.date(2027, 2, 1)).state == "active"


def test_tts_1_dated_snapshots_and_gpt5_snapshots_match():
    assert status_for("openai", "tts-1-1106", today=dt.date(2027, 1, 6)).state == "shutdown"
    for m in ("gpt-5-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"):
        assert status_for("openai", m, today=dt.date(2026, 12, 11)).state == "shutdown"
    assert status_for("openai", "gpt-5", today=dt.date(2026, 12, 11)).state == "active"


def test_openai_stt_default_is_flagged_from_oct_2():
    assert status_for("openai", "gpt-4o-transcribe", today=TODAY).state == "warning"
    assert status_for("openai", "whisper-1", today=dt.date(2026, 10, 1)).state == "active"
