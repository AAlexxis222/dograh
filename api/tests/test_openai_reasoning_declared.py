"""OpenAI reasoning is declared on the wire, never left to a library default.

VOZ-BUG-03: pipecat's Responses service sends ``effort: "none"`` when nobody
sets it. The effort matrix (``openai_efforts``) declares every impact knob per
(model, api); the characterization table pins what chat/completions sends
today so the refactor exposes the behaviour instead of changing it.
"""

import pytest
from openai import NOT_GIVEN

from api.services.pipecat.service_factory import create_llm_service_from_provider
from api.tests.service_tuning._transport import chat_payload

_IMPACT_KNOBS = ("temperature", "reasoning_effort", "verbosity")
_GPT5_TODAY = {"reasoning_effort": "minimal", "verbosity": "low"}
_AZURE = {"endpoint": "https://example.openai.azure.com"}


def _impact_knobs(payload: dict) -> dict:
    return {
        name: payload[name]
        for name in _IMPACT_KNOBS
        if name in payload and payload[name] is not NOT_GIVEN
    }


# (provider, model, ctor kwargs) -> what chat/completions sends before the
# effort matrix exists. Green before and after the refactor.
TODAY_CHAT = [
    ("openai", "gpt-4.1", {}, {"temperature": 0.1}),
    ("openai", "gpt-4.1-mini", {}, {"temperature": 0.1}),
    ("openai", "gpt-5", {}, _GPT5_TODAY),
    ("openai", "gpt-5-mini", {}, _GPT5_TODAY),
    ("openai", "gpt-5-nano", {}, _GPT5_TODAY),
    ("openai", "gpt-5.1", {}, _GPT5_TODAY),
    ("openai", "gpt-5.4-mini", {}, _GPT5_TODAY),
    ("openai", "gpt-5.6-luna", {}, _GPT5_TODAY),
    ("openai", "gpt-5-chat-latest", {}, {"temperature": 0.1}),
    ("atlascloud", "gpt-5-mini", {}, _GPT5_TODAY),
    ("openrouter", "openai/gpt-5-mini", {}, {}),
    ("openrouter", "openai/gpt-4.1", {}, {"temperature": 0.1}),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct", {}, {"temperature": 0.1}),
    ("azure", "gpt-5", _AZURE, {}),
    ("azure", "gpt-4.1", _AZURE, {"temperature": 0.1}),
    ("dograh", "gpt-5-mini", {}, {}),
    ("dograh", "gpt-4.1", {}, {}),
]


@pytest.mark.parametrize("provider, model, kwargs, expected", TODAY_CHAT)
def test_chat_completions_keeps_todays_impact_knobs(provider, model, kwargs, expected):
    llm = create_llm_service_from_provider(
        provider=provider, model=model, api_key="k", **kwargs
    )
    assert _impact_knobs(chat_payload(llm)) == expected
