"""``service_tuning`` on the OpenAI chat-completions LLM, asserted on the wire.

Today's gpt-5 special case (two ``extra`` keys and no temperature) becomes a
declared default here: with no tuning the payload is byte-for-byte what the
control tests pin, and ``options.reasoning``/``options.verbosity`` replace the
two extras instead of adding a second mechanism.
"""

import pytest
from fastapi import HTTPException
from openai import NOT_GIVEN
from pydantic import ValidationError

from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.pipecat.service_factory import (
    _gpt5_chat_plan,
    create_llm_service_from_provider,
)
from api.services.pipecat.service_tuning import tuning_for
from api.tests.service_tuning._transport import chat_payload


def _llm(model, tuning=None):
    return create_llm_service_from_provider(
        provider="openai", model=model, api_key="k", tuning=tuning
    )


def test_gpt5_default_payload_reproduces_todays_extras_exactly():
    p = chat_payload(_llm("gpt-5-mini"))
    assert (
        p["reasoning_effort"] == "minimal"
        and p["verbosity"] == "low"
        and p["temperature"] is NOT_GIVEN
    )


def test_gpt41_default_payload_reproduces_todays_temperature():
    p = chat_payload(_llm("gpt-4.1"))
    assert p["temperature"] == 0.1 and "reasoning_effort" not in p


def test_all_temperature_applies_to_openai_and_provider_wins():
    assert (
        chat_payload(
            _llm("gpt-4.1", {"llm": {"_all": {"settings": {"temperature": 0.5}}}})
        )["temperature"]
        == 0.5
    )
    assert (
        chat_payload(
            _llm(
                "gpt-4.1",
                {
                    "llm": {
                        "_all": {"settings": {"temperature": 0.5}},
                        "openai": {"settings": {"temperature": 0.9}},
                    }
                },
            )
        )["temperature"]
        == 0.9
    )


def test_gpt5_reasoning_and_verbosity_options_replace_the_extras():
    p = chat_payload(
        _llm(
            "gpt-5.1",
            {
                "llm": {
                    "openai": {
                        "options": {
                            "reasoning": {"effort": "low"},
                            "verbosity": "medium",
                        }
                    }
                }
            },
        )
    )
    assert p["reasoning_effort"] == "low" and p["verbosity"] == "medium"


def test_gpt5_with_an_explicit_temperature_fails_the_run_start_by_name():
    # gpt-5 reasoning models reject ``temperature`` over chat/completions and
    # ``build_chat_completion_params`` sends it verbatim (openai/base_llm.py
    # :356-358). Dropping it with a warning would start the run and silently
    # run it differently from what the document says (§1.2), so the knob
    # fails the run start with the model in the message.
    with pytest.raises(HTTPException) as info:
        _llm("gpt-5.1", {"llm": {"_all": {"settings": {"temperature": 0.3}}}})
    assert info.value.status_code == 400
    assert info.value.detail == (
        "llm temperature is not accepted by gpt-5.1 over chat/completions; "
        "remove llm.*.settings.temperature or choose another model"
    )


def test_gpt5_translates_the_completion_cap():
    # The cap is a lossless translation: gpt-5 takes it only as
    # ``max_completion_tokens``.
    p = chat_payload(
        _llm(
            "gpt-5.1",
            {"llm": {"openai": {"settings": {"max_tokens": 400, "top_p": 0.8}}}},
        )
    )
    assert p["max_tokens"] is NOT_GIVEN and p["max_completion_tokens"] == 400
    assert p["top_p"] == 0.8
    assert p["reasoning_effort"] == "minimal" and p["verbosity"] == "low"


def test_gpt5_chat_takes_temperature_and_no_reasoning_extras():
    # ``gpt-5-chat*`` is the non-reasoning member of the family: it accepts
    # temperature and rejects ``reasoning_effort``/``verbosity``, so the rule
    # is keyed to the model, not to the ``gpt-5`` prefix (#15).
    p = chat_payload(
        _llm(
            "gpt-5-chat-latest",
            {"llm": {"_all": {"settings": {"temperature": 0.3}}}},
        )
    )
    assert p["temperature"] == 0.3
    assert "reasoning_effort" not in p and "verbosity" not in p


def test_gpt5_keeps_an_explicit_max_completion_tokens():
    p = chat_payload(
        _llm(
            "gpt-5.1",
            {
                "llm": {
                    "openai": {
                        "settings": {"max_tokens": 400, "max_completion_tokens": 900}
                    }
                }
            },
        )
    )
    assert p["max_completion_tokens"] == 900 and p["max_tokens"] is NOT_GIVEN


def test_gpt41_still_sends_temperature_and_max_tokens_verbatim():
    p = chat_payload(
        _llm(
            "gpt-4.1",
            {
                "llm": {
                    "_all": {"settings": {"temperature": 0.3}},
                    "openai": {"settings": {"max_tokens": 400}},
                }
            },
        )
    )
    assert p["temperature"] == 0.3 and p["max_tokens"] == 400
    assert p["max_completion_tokens"] is NOT_GIVEN


def test_gpt5_translation_never_mutates_the_plan():
    # EMPTY_PLAN is a shared singleton and every plan dict is read again by
    # the next branch, so the translation has to return a copy.
    plan = tuning_for(
        {"llm": {"openai": {"settings": {"max_tokens": 400, "top_p": 0.8}}}},
        "llm",
        "openai",
    )
    translated = _gpt5_chat_plan(plan, "gpt-5.1")
    assert plan.settings == {"max_tokens": 400, "top_p": 0.8}
    assert translated.settings == {"max_completion_tokens": 400, "top_p": 0.8}


def test_max_tokens_and_top_p_reach_the_payload():
    p = chat_payload(
        _llm(
            "gpt-4.1",
            {"llm": {"openai": {"settings": {"max_tokens": 300, "top_p": 0.8}}}},
        )
    )
    assert p["max_tokens"] == 300 and p["top_p"] == 0.8


def test_openai_extra_is_never_accepted():
    with pytest.raises(ValidationError, match="llm.openai.settings.extra"):
        WorkflowConfigurationDefaults.model_validate(
            {
                "service_tuning": {
                    "llm": {"openai": {"settings": {"extra": {"model": "x"}}}}
                }
            }
        )


def test_responses_api_option_is_rejected_until_the_deferral_port_lands():
    # Task 12 flips this to a real service; until then the knob is a named 422.
    with pytest.raises(
        ValidationError, match="llm.openai.options.api: responses is not available"
    ):
        WorkflowConfigurationDefaults.model_validate(
            {"service_tuning": {"llm": {"openai": {"options": {"api": "responses"}}}}}
        )


def test_reasoning_options_outside_gpt5_are_dropped_with_a_logged_warning():
    from loguru import logger

    records = []
    sink = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    try:
        p = chat_payload(
            _llm(
                "gpt-4.1",
                {"llm": {"openai": {"options": {"reasoning": {"effort": "low"}}}}},
            )
        )
    finally:
        logger.remove(sink)
    assert "reasoning_effort" not in p
    assert any(
        w.startswith("service_tuning:")
        and "llm.openai.options.reasoning" in w
        and "gpt-4.1" in w
        for w in records
    )
