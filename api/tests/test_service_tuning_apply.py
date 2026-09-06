"""The applier merges ``_all`` under the provider section and hands the result
to ``Settings.from_mapping`` as a delta over the factory's current kwargs."""

from pipecat.services.deepgram.flux.base import DeepgramFluxSTTSettings
from pipecat.services.settings import NOT_GIVEN

from api.services.pipecat.service_tuning import (
    EMPTY_PLAN,
    build_settings,
    llm_tuning_applies,
    tuning_for,
)


def test_tuning_for_merges_all_then_provider():
    doc = {
        "llm": {
            "_all": {"settings": {"temperature": 0.3, "max_tokens": 100}},
            "openai": {
                "settings": {"temperature": 0.9},
                "options": {"verbosity": "low"},
            },
        }
    }
    plan = tuning_for(doc, "llm", "openai")
    assert plan.settings == {"temperature": 0.9, "max_tokens": 100}
    assert plan.options == {"verbosity": "low"}
    assert tuning_for(doc, "llm", "groq").settings == {
        "temperature": 0.3,
        "max_tokens": 100,
    }
    assert tuning_for(None, "stt", "deepgram") == EMPTY_PLAN


def test_build_settings_applies_delta_and_keeps_explicit_null():
    plan = tuning_for(
        {
            "stt": {
                "deepgram": {
                    "settings": {"eager_eot_threshold": None, "numerals": True}
                }
            }
        },
        "stt",
        "deepgram",
    )
    s = build_settings(
        DeepgramFluxSTTSettings,
        {
            "model": "flux-general-multi",
            "eot_threshold": 0.7,
            "eager_eot_threshold": 0.5,
            "keyterm": [],
        },
        plan,
    )
    assert (
        s.eager_eot_threshold is None and s.numerals is True and s.eot_threshold == 0.7
    )
    assert s.extra == {}
    assert s.min_confidence is NOT_GIVEN


def test_build_settings_without_plan_is_identical_to_literal_construction():
    base = {
        "model": "flux-general-multi",
        "eot_timeout_ms": 3000,
        "eot_threshold": 0.7,
        "eager_eot_threshold": 0.5,
        "keyterm": ["a"],
    }
    assert build_settings(DeepgramFluxSTTSettings, base, EMPTY_PLAN) == (
        DeepgramFluxSTTSettings(**base)
    )


def test_llm_scope_defaults_to_conversation_only():
    doc = {"llm": {"_all": {"settings": {"temperature": 0.2}}}}
    assert llm_tuning_applies(doc, "conversation") is True
    assert llm_tuning_applies(doc, "extraction") is False
    assert (
        llm_tuning_applies({**doc, "scope": {"extraction": True}}, "extraction") is True
    )
    assert llm_tuning_applies(None, "conversation") is False
