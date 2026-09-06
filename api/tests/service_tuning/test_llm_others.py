"""``service_tuning`` on the non-OpenAI LLM branches, asserted on the wire.

Google is the one provider whose knobs never reach a chat-completions payload;
its wire equivalent is ``_build_generation_params`` (google/llm.py:364-404),
the dict handed to ``GenerateContentConfig``.
"""

from types import SimpleNamespace

from openai import NOT_GIVEN

from api.services.pipecat.service_factory import (
    create_llm_service,
    create_llm_service_from_provider,
)
from api.tests.service_tuning._transport import chat_payload


def test_google_thinking_reaches_generation_params():
    llm = create_llm_service_from_provider(
        provider="google",
        model="gemini-2.5-flash",
        api_key="k",
        tuning={
            "llm": {"google": {"settings": {"thinking": {"thinking_budget": 512}}}}
        },
    )
    params = llm._build_generation_params()
    assert params["thinking_config"]["thinking_budget"] == 512
    assert params["temperature"] == 0.1


def test_sarvam_reasoning_effort_reaches_the_payload():
    llm = create_llm_service_from_provider(
        provider="sarvam",
        model="sarvam-105b",
        api_key="k",
        temperature=0.5,
        tuning={"llm": {"sarvam": {"settings": {"reasoning_effort": "high"}}}},
    )
    assert chat_payload(llm)["reasoning_effort"] == "high"


def test_dograh_llm_gets_temperature_only_when_tuned():
    llm = create_llm_service_from_provider(
        provider="dograh",
        model="default",
        api_key="k",
        tuning={"llm": {"_all": {"settings": {"temperature": 0.2}}}},
    )
    assert chat_payload(llm)["temperature"] == 0.2
    assert (
        chat_payload(
            create_llm_service_from_provider(
                provider="dograh", model="default", api_key="k"
            )
        )["temperature"]
        is NOT_GIVEN
    )


def test_llm_tuning_does_not_apply_to_extraction_role_by_default():
    llm = create_llm_service(
        SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai", model="gpt-4.1", api_key="k", base_url=None
            )
        ),
        tuning={"llm": {"_all": {"settings": {"temperature": 0.9}}}},
        role="extraction",
    )
    assert chat_payload(llm)["temperature"] == 0.1
