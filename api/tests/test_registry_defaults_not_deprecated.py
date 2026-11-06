"""VOZ-AC-B1-81: no registry default points to a model with a past date or legacy/deprecated status."""

import datetime as dt
import inspect
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from api.services.capabilities.deprecations import (
    MODEL_KINDS,
    blocking_patterns,
    load_deprecations,
    status_for,
)
from api.services.configuration import registry

# A fixed guard date, not today: after every retirement up to the 2027-01-20 realtime wave, so a default that is
# scheduled to retire fails here weeks ahead instead of passing until the day it breaks.
GUARD_DATE = dt.date(2027, 2, 1)

# Defaults known to retire before GUARD_DATE: (provider, model) -> (follow-up that moves it, why it stays for now).
# test_every_exception_is_still_needed fails once an entry stops being needed, so the list cannot rot.
EXCEPTIONS = {
    ("openai", "gpt-4.1"): (
        "VOZ-FU-N0-01b-llm-default",
        "OpenAI retires it on 2027-01-14; the default moves together with the voicemail and UI fallbacks",
    ),
    ("openrouter", "openai/gpt-4.1"): (
        "VOZ-FU-N0-01b-llm-default",
        "legacy proxy of the same OpenAI model; it moves with the OpenAI default",
    ),
    ("lmnt", "aurora"): (
        "VOZ-FU-N0-01b-refuse-on-save",
        "LMNT shut down; the class stays, outside REGISTRY, only so stored configs still load",
    ),
}


def _provider_name(provider) -> str:
    """A registry provider key (enum member or plain string) as the string deprecations.yaml uses."""
    return str(getattr(provider, "value", provider))


def _provider(cls) -> str:
    return _provider_name(cls.model_fields["provider"].default)


def _configs():
    for name, cls in inspect.getmembers(registry, inspect.isclass):
        if (
            issubclass(cls, BaseModel)
            and cls.__module__ == registry.__name__
            and "model" in cls.model_fields
            and "provider" in cls.model_fields
        ):
            yield name, cls


_DEFAULTS_DIR = Path(registry.__file__).parent


def _service_deprecated_as_a_whole(cls) -> bool:
    """True when deprecations.yaml retires the class's whole service (``kind: service``, no 1:1 replacement)."""
    provider = _provider(cls)
    service_types = {
        t.name.lower()
        for t, by_provider in registry.REGISTRY.items()
        if by_provider.get(cls.model_fields["provider"].default) is cls
    }
    return any(
        e["kind"] == "service"
        and e["provider"] == provider
        and e["model_or_feature"] in service_types
        for e in load_deprecations()
    )


@pytest.mark.parametrize("name,cls", list(_configs()))
def test_registry_defaults_not_deprecated(name, cls):
    if _service_deprecated_as_a_whole(cls):
        # Kept only for existing workflows (VOZ-AC-B1-83): it must not be a default anywhere.
        offenders = [
            p.name
            for p in _DEFAULTS_DIR.glob("defaults*.py")
            if name in p.read_text(encoding="utf-8")
        ]
        assert not offenders, (
            f"{name} is deprecated as a service but is a default in {offenders}"
        )
        return
    provider = _provider(cls)
    model = cls.model_fields["model"].default
    if (provider, model) in EXCEPTIONS:
        return
    status = status_for(provider, model, today=GUARD_DATE)
    assert status.state in {"active", "warning_price"}, (
        f"{name}: default model {provider}/{model} is {status.state} ({status.reason}); hint: {status.hint}"
    )


@pytest.mark.parametrize("key", list(EXCEPTIONS))
def test_every_exception_is_still_needed(key):
    """An exception must name a current default that really is retiring at GUARD_DATE."""
    defaults = {
        (_provider(cls), cls.model_fields["model"].default) for _, cls in _configs()
    }
    assert key in defaults, (
        f"{key} is no longer a registry default: drop it from EXCEPTIONS"
    )
    assert status_for(*key, today=GUARD_DATE).state not in {
        "active",
        "warning_price",
    }, f"{key} is no longer retiring at {GUARD_DATE}: drop it from EXCEPTIONS"


@pytest.mark.parametrize("model", registry.LMNT_TTS_MODELS)
def test_every_lmnt_model_is_shutdown_for_status_and_text_regex(model):
    """LMNT shut down as a whole: a stored config holds one of these ids, so each must fail the sweep both ways."""
    retired = dt.date(2026, 11, 5)
    assert status_for("lmnt", model, today=retired).state == "shutdown"
    assert model in blocking_patterns(today=retired)


def test_azure_llm_model_is_a_deployment_name_and_is_never_matched_as_a_model_id():
    """AzureLLMService.model names a deployment, not the model it runs: an id-based entry would flag the wrong rows."""
    assert (
        "deployment name" in registry.AzureLLMService.model_fields["model"].description
    )
    for today in (dt.date(2027, 4, 14), dt.date(2028, 1, 1)):
        assert status_for("azure", "gpt-4.1-mini", today=today).state == "active"


# The offered lists are checked at the next retirement wave, not at GUARD_DATE (RUL-r1-8): no model picker may still
# offer an id that is shut down by NEXT_WAVE_DATE, while the ids that shut down between it and GUARD_DATE stay
# offered until the follow-up named in OFFERED_UNTIL_FOLLOW_UP removes them.
NEXT_WAVE_DATE = dt.date(2026, 12, 15)

# Offered ids shut down at GUARD_DATE that stay listed for now: (provider, model) -> follow-up that removes them.
# test_offered_until_follow_up_is_exact fails when an id is missing here or no longer needs to be.
OFFERED_UNTIL_FOLLOW_UP = {
    ("openai", "gpt-4.1"): "VOZ-FU-N0-01b-offered-lists-2027",
    ("openai", "gpt-4.1-mini"): "VOZ-FU-N0-01b-offered-lists-2027",
    ("openai", "gpt-4o-mini-tts"): "VOZ-FU-N0-01b-offered-lists-2027",
    ("azure_realtime", "gpt-realtime"): "VOZ-FU-N0-01b-offered-lists-2027",
    ("azure_realtime", "gpt-realtime-mini"): "VOZ-FU-N0-01b-offered-lists-2027",
}


def _offered_models():
    """(service type, provider, model) for every id a registered class offers in its model picker."""
    for service_type, by_provider in registry.REGISTRY.items():
        for provider, cls in by_provider.items():
            field = cls.model_fields.get("model")
            extra = field.json_schema_extra if field else None
            for model in extra.get("examples", ()) if isinstance(extra, dict) else ():
                yield service_type.name.lower(), _provider_name(provider), model


@pytest.mark.parametrize(
    "provider,model",
    [
        pytest.param(p, m, id=f"{t}:{p}:{m}")
        for t, p, m in _offered_models()
        if (p, m) not in EXCEPTIONS
    ],
)
def test_offered_models_are_not_shutdown_at_the_next_wave(provider, model):
    status = status_for(provider, model, today=NEXT_WAVE_DATE)
    assert status.state != "shutdown", (
        f"{provider}/{model} is still offered: {status.reason}"
    )


def test_offered_until_follow_up_is_exact():
    """OFFERED_UNTIL_FOLLOW_UP names exactly the offered ids that are shut down at GUARD_DATE."""
    retiring = {
        (p, m)
        for _, p, m in _offered_models()
        if status_for(p, m, today=GUARD_DATE).state == "shutdown"
    }
    named = set(OFFERED_UNTIL_FOLLOW_UP)
    assert not retiring - named, (
        f"offered and shut down at {GUARD_DATE}, name a follow-up: {sorted(retiring - named)}"
    )
    assert not named - retiring, (
        f"no longer offered or no longer retiring: drop {sorted(named - retiring)}"
    )


def _slot_class(kind, provider):
    """The registered class for a deprecations.yaml model kind and provider, or None."""
    for service_type, by_provider in registry.REGISTRY.items():
        for p, cls in by_provider.items():
            if service_type.name.lower() == kind and _provider_name(p) == provider:
                return cls
    return None


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(e, id=f"{e['provider']}:{e['replacement']}")
        for e in load_deprecations()
        if e.get("replacement")
    ],
)
def test_every_replacement_is_an_active_model_of_the_same_slot(entry):
    """A hint must not send an operator to another slot's model, nor to a model that is retiring itself."""
    kind, provider = entry["kind"], entry["provider"]
    assert kind in MODEL_KINDS, f"only model entries can name a replacement: {entry}"
    date = dt.date.fromisoformat(str(entry["date"])) if entry["date"] else GUARD_DATE
    cls = _slot_class(kind, provider)
    extra = cls.model_fields["model"].json_schema_extra if cls else None
    for model in (
        t for t in re.findall(r"[^\s()]+", entry["replacement"]) if t != "or"
    ):
        status = status_for(provider, model, today=date)
        assert status.state == "active", (
            f"replacement {provider}/{model} is {status.state} on {date}"
        )
        other_slots = {t for t, _, m in _offered_models() if m == model and t != kind}
        assert not other_slots, (
            f"replacement {model} is offered as {other_slots}, not as {kind}"
        )
        if isinstance(extra, dict) and not extra.get("allow_custom_input"):
            offered = [*extra.get("examples", ()), cls.model_fields["model"].default]
            assert model in offered, (
                f"replacement {model} cannot be picked in the {provider} {kind} slot: {offered}"
            )


@pytest.mark.parametrize("model", ["openai/gpt-4.1", "openai/gpt-4.1-mini"])
def test_openrouter_proxies_of_the_retiring_gpt_4_1_family_are_flagged(model):
    """status_for filters by provider, so OpenAI's own entry never reaches the OpenRouter slugs of the same models."""
    assert model in registry.OPENROUTER_MODELS
    assert status_for("openrouter", model, today=GUARD_DATE).state == "legacy"


def test_price_effect_only_warns():
    s = status_for("elevenlabs", "eleven_v4_turbo", today=dt.date(2026, 10, 13))
    assert s.state == "warning_price"


def test_shutdown_is_blocking_from_24h_before_date():
    assert (
        status_for("cartesia", "sonic-2", today=dt.date(2026, 10, 19)).state
        == "shutdown"
    )


def test_every_entry_has_source_and_read_at():
    for e in load_deprecations():
        assert e["source_url"].startswith("http") and e["read_at"], e


def test_legacy_entry_has_no_date_and_is_legacy():
    s = status_for("grok_realtime", "grok-voice-think-fast-1.0", today=GUARD_DATE)
    assert s.state == "legacy" and "grok-voice-latest" in s.hint


@pytest.mark.parametrize(
    "model",
    [
        "sonic-2",
        "sonic-2-2025-06-11",
        "sonic-turbo",
        "sonic-turbo-2025-06-04",
        "sonic-3-2025-10-27",
    ],
)
def test_cartesia_sunset_models_and_dated_snapshots_are_shutdown(model):
    assert (
        status_for("cartesia", model, today=dt.date(2026, 10, 20)).state == "shutdown"
    )


@pytest.mark.parametrize(
    "model",
    [
        "sonic",
        "sonic-english",
        "sonic-multilingual",
        "sonic-2024-12-12",
        "sonic-2024-10-19",
    ],
)
def test_cartesia_models_already_sunset_in_june_are_shutdown_now(model):
    assert status_for("cartesia", model, today=GUARD_DATE).state == "shutdown"


@pytest.mark.parametrize(
    "model",
    [
        "sonic-3",
        "sonic-3-2026-01-12",
        "sonic-3.5",
        "sonic-3.6",
        "sonic-3.6-2026-08-27",
        "sonic-preview",
    ],
)
def test_supported_cartesia_models_stay_active(model):
    assert status_for("cartesia", model, today=dt.date(2026, 10, 25)).state == "active"


@pytest.mark.parametrize("provider", ["openai_realtime", "azure_realtime"])
@pytest.mark.parametrize(
    "model",
    [
        "gpt-realtime",
        "gpt-realtime-2025-08-28",
        "gpt-realtime-mini",
        "gpt-realtime-mini-2025-12-15",
        "gpt-4o-realtime-preview",
    ],
)
def test_realtime_retirement_uses_registry_provider_keys(provider, model):
    assert status_for(provider, model, today=dt.date(2027, 1, 20)).state == "shutdown"


@pytest.mark.parametrize(
    "model", ["gpt-realtime-2", "gpt-realtime-2.1", "gpt-realtime-2.1-mini"]
)
def test_realtime_replacements_are_not_flagged(model):
    assert (
        status_for("openai_realtime", model, today=dt.date(2027, 2, 1)).state
        == "active"
    )


def test_openai_provider_key_no_longer_carries_realtime_entry():
    assert (
        status_for("openai", "gpt-realtime", today=dt.date(2027, 2, 1)).state
        == "active"
    )


def test_tts_1_dated_snapshots_and_gpt5_snapshots_match():
    assert (
        status_for("openai", "tts-1-1106", today=dt.date(2027, 1, 6)).state
        == "shutdown"
    )
    for m in ("gpt-5-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"):
        assert status_for("openai", m, today=dt.date(2026, 12, 11)).state == "shutdown"
    assert status_for("openai", "gpt-5", today=dt.date(2026, 12, 11)).state == "active"


def test_openai_stt_default_is_flagged_from_oct_2():
    assert (
        status_for("openai", "gpt-4o-transcribe", today=GUARD_DATE).state == "warning"
    )
    assert (
        status_for("openai", "whisper-1", today=dt.date(2026, 10, 1)).state == "active"
    )


def test_dated_mini_transcribe_snapshot_keeps_its_own_date_over_the_broad_glob():
    s = status_for(
        "openai", "gpt-4o-mini-transcribe-2025-03-20", today=dt.date(2027, 1, 19)
    )
    assert s.date == dt.date(2027, 1, 20)


def _literal_ids(entry):
    ids = entry["model_or_feature"]
    ids = ids if isinstance(ids, list) else [ids]
    return [m for m in ids if not any(c in m for c in "*?[")]


@pytest.mark.parametrize(
    "provider,model,entry",
    [
        pytest.param(e["provider"], m, e, id=f"{e['provider']}:{m}")
        for e in load_deprecations()
        for m in _literal_ids(e)
    ],
)
def test_no_entry_is_shadowed_by_an_earlier_one(provider, model, entry):
    """status_for takes the first match: every literal id must resolve to the entry that lists it."""
    if entry["date"] is None:
        status = status_for(provider, model, today=GUARD_DATE)
        assert status.state == "legacy"
    else:
        date = dt.date.fromisoformat(str(entry["date"]))
        status = status_for(provider, model, today=date)
        assert status.date == date
    if entry.get("replacement"):
        assert entry["replacement"] in status.hint


def test_assemblyai_offered_pro_models_are_recognised_by_pinned_pipecat():
    from pipecat.services.assemblyai.stt import is_u3_pro_model

    default = registry.AssemblyAISTTConfiguration.model_fields["model"].default
    assert is_u3_pro_model(default), (
        f"{default} is not a U3 Pro model for the pinned pipecat"
    )
    for m in registry.ASSEMBLYAI_STT_MODELS:
        if m.startswith(("universal-3", "u3")):
            assert is_u3_pro_model(m), (
                f"{m} is offered but the pinned pipecat treats it as non-Pro"
            )
