"""``service_tuning`` on the OpenAI chat-completions LLM, asserted on the wire.

Today's gpt-5 special case (two ``extra`` keys and no temperature) becomes a
declared default here: with no tuning the payload is byte-for-byte what the
control tests pin, and ``options.reasoning``/``options.verbosity`` replace the
two extras instead of adding a second mechanism.
"""

import pytest
from openai import NOT_GIVEN
from pydantic import ValidationError

from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.pipecat.service_factory import create_llm_service_from_provider
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


def test_gpt5_temperature_is_sent_only_when_set():
    p = chat_payload(
        _llm("gpt-5.1", {"llm": {"openai": {"settings": {"temperature": 1.0}}}})
    )
    assert p["temperature"] == 1.0 and p["reasoning_effort"] == "minimal"


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
