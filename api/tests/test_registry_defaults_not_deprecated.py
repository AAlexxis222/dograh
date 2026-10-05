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
    "GoogleRealtimeLLMConfiguration",
    "GoogleVertexRealtimeLLMConfiguration",
    "GrokRealtimeLLMConfiguration",
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
