"""OpenAI reasoning is declared on the wire, never left to a library default.

VOZ-BUG-03: pipecat's Responses service sends ``effort: "none"`` when nobody
sets it. The effort matrix (``openai_efforts``) declares every impact knob per
(model, api); the characterization table pins what chat/completions sends
today so the refactor exposes the behaviour instead of changing it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from openai import NOT_GIVEN
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
)
from pipecat.frames.frames import BotStoppedSpeakingFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

from api.services.pipecat.openai_efforts import EffortError, wire_params
from api.services.pipecat.openai_responses import DograhOpenAIResponsesLLMService
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


# --- the effort matrix (plan Step 1, binding) -------------------------------


@pytest.mark.parametrize(
    "model", ["gpt-6-luna", "gpt-6-sol", "gpt-6-astra", "gpt-6.1-sol"]
)
def test_reasoning_effort_always_on_the_wire(model):
    params = wire_params(model=model, requested_effort=None)
    assert params["reasoning"]["effort"] in {"low", "medium", "high", "max", "minimal"}
    assert params["reasoning"]["effort"] != "none"


def test_none_is_rejected_where_the_model_forbids_it():
    with pytest.raises(EffortError) as e:
        wire_params(model="gpt-6-astra", requested_effort="none")
    assert e.value.code == "knob_out_of_range" and e.value.hint


def test_temperature_only_with_effort_none():
    assert "temperature" not in wire_params(
        model="gpt-6-luna", requested_effort="medium"
    )


def test_gpt5_chat_keeps_todays_behaviour():  # VOZ-AC-B1-47 «exponer ≠ cambiar»
    p = wire_params(model="gpt-5", requested_effort=None, api="chat_completions")
    assert p == {"reasoning_effort": "minimal", "verbosity": "low"}


def test_responses_gate_is_on():
    from api.services.pipecat.service_tuning_specs import RESPONSES_API_AVAILABLE

    assert RESPONSES_API_AVAILABLE is True


# --- the rest of the matrix (VOZ-AC-B1-47) ----------------------------------


@pytest.mark.parametrize(
    "model, effort",
    [
        ("gpt-6.1-sol", "none"),
        ("gpt-6-luna", "minimal"),
        ("gpt-5.1", "minimal"),
        ("gpt-5", "max"),
        ("gpt-5.4-mini", "max"),
        ("gpt-4.1", "low"),
    ],
)
def test_efforts_outside_the_row_fail_with_the_b0_28_shape(model, effort):
    with pytest.raises(EffortError) as e:
        wire_params(model=model, requested_effort=effort)
    err = e.value
    assert err.code == "knob_out_of_range"
    assert err.where == "llm.openai.options.reasoning.effort"
    assert model in err.reason and effort in err.reason and err.hint


@pytest.mark.parametrize(
    "model, effort",
    [("gpt-5-mini", "minimal"), ("gpt-5.6-luna", "max"), ("gpt-6-sol", "max")],
)
def test_minimal_and_max_live_only_where_the_docs_put_them(model, effort):
    assert wire_params(model=model, requested_effort=effort) == {
        "reasoning": {"effort": effort}
    }


def test_effort_none_carries_todays_temperature():
    assert wire_params(model="gpt-6-luna", requested_effort="none") == {
        "reasoning": {"effort": "none"},
        "temperature": 0.1,
    }


def test_a_model_outside_the_matrix_cannot_use_responses():
    with pytest.raises(EffortError) as e:
        wire_params(model="gpt-7-nova", requested_effort=None)
    assert e.value.code == "api_not_supported" and e.value.hint


# --- chat/completions cases where today's effort was an implicit API default --
# Today these models get ``temperature: 0.1`` and no effort, so the API's
# documented default (``medium``) decides; it is now declared, and the
# temperature the docs reject with any effort but ``none`` is no longer sent.
IMPLICIT_CHAT = [
    ("openai", "gpt-6-luna", None, {"reasoning_effort": "medium"}),
    ("openai", "gpt-6-sol", None, {"reasoning_effort": "medium"}),
    ("openai", "gpt-6-astra", None, {"reasoning_effort": "medium"}),
    ("openai", "gpt-6.1-sol", None, {"reasoning_effort": "medium"}),
    (
        "openai",
        "gpt-6-luna",
        {"llm": {"openai": {"options": {"reasoning": {"effort": "none"}}}}},
        {"reasoning_effort": "none", "temperature": 0.1},
    ),
    ("openrouter", "openai/gpt-6-luna", None, {}),
]


@pytest.mark.parametrize("provider, model, tuning, expected", IMPLICIT_CHAT)
def test_chat_completions_declares_what_was_implicit(provider, model, tuning, expected):
    llm = create_llm_service_from_provider(
        provider=provider, model=model, api_key="k", tuning=tuning
    )
    assert _impact_knobs(chat_payload(llm)) == expected


# --- VOZ-AT-B1-14 on the wire: Responses through a fake transport ------------


def _responses_tuning(**options):
    return {"llm": {"openai": {"options": {"api": "responses", **options}}}}


def _fake_transport(service, events=()):
    """Replace the HTTP client: capture each request, stream ``events`` back."""
    sent: list[dict] = []

    async def create(**params):
        sent.append(params)

        async def stream():
            for event in events:
                yield event

        return stream()

    service._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    service.push_frame = AsyncMock()
    return sent


async def _one_turn(service, events=()):
    sent = _fake_transport(service, events)
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "hola"}])
    )
    return sent[0]


@pytest.mark.parametrize(
    "model", ["gpt-6-luna", "gpt-6-sol", "gpt-6-astra", "gpt-6.1-sol"]
)
async def test_responses_wire_declares_medium_and_never_none(model):
    llm = create_llm_service_from_provider(
        provider="openai", model=model, api_key="k", tuning=_responses_tuning()
    )
    assert isinstance(llm, DograhOpenAIResponsesLLMService)
    sent = await _one_turn(llm)
    assert sent["reasoning"] == {"effort": "medium"}
    assert "temperature" not in sent and sent["store"] is False


async def test_responses_wire_carries_a_requested_none_with_temperature():
    llm = create_llm_service_from_provider(
        provider="openai",
        model="gpt-6-luna",
        api_key="k",
        tuning=_responses_tuning(reasoning={"effort": "none"}),
    )
    sent = await _one_turn(llm)
    assert sent["reasoning"] == {"effort": "none"} and sent["temperature"] == 0.1


async def test_responses_wire_carries_summary_and_verbosity():
    llm = create_llm_service_from_provider(
        provider="openai",
        model="gpt-6-luna",
        api_key="k",
        tuning=_responses_tuning(
            reasoning={"effort": "high", "summary": "concise"}, verbosity="low"
        ),
    )
    sent = await _one_turn(llm)
    assert sent["reasoning"] == {"effort": "high", "summary": "concise"}
    assert sent["text"] == {"verbosity": "low"}


def _responses_llm(effort, **settings):
    return create_llm_service_from_provider(
        provider="openai",
        model="gpt-6-luna",
        api_key="k",
        tuning={
            "llm": {
                "openai": {
                    "options": {"api": "responses", "reasoning": {"effort": effort}},
                    "settings": settings,
                }
            }
        },
    )


async def test_responses_wire_takes_the_cap_and_an_explicit_temperature_with_none():
    sent = await _one_turn(_responses_llm("none", temperature=0.3, max_tokens=400))
    assert sent["temperature"] == 0.3 and sent["max_output_tokens"] == 400


@pytest.mark.parametrize(
    "settings, where",
    [
        ({"temperature": 0.3}, None),
        ({"seed": 7}, "llm.openai.settings.seed"),
    ],
)
def test_responses_refuses_knobs_it_would_not_honour(settings, where):
    with pytest.raises(HTTPException) as info:
        _responses_llm("medium", **settings)
    detail = info.value.detail
    if where is None:  # temperature next to an effort other than none
        assert "not accepted by gpt-6-luna over the Responses API" in detail
    else:
        assert detail["code"] == "knob_not_supported" and detail["where"] == where


def test_factory_names_the_out_of_range_effort():
    with pytest.raises(HTTPException) as info:
        create_llm_service_from_provider(
            provider="openai",
            model="gpt-6-astra",
            api_key="k",
            tuning=_responses_tuning(reasoning={"effort": "none"}),
        )
    detail = info.value.detail
    assert info.value.status_code == 400 and detail["code"] == "knob_out_of_range"
    assert all(detail[k] for k in ("code", "where", "reason", "hint"))


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6.1-sol"])
async def test_unconfigured_reasoning_is_declared_not_disabled(model):
    # Built without the factory: pipecat would send ``effort: "none"`` here
    # (a 400 on gpt-6.1-sol); the subclass declares the matrix default.
    llm = DograhOpenAIResponsesLLMService(
        api_key="k", settings=DograhOpenAIResponsesLLMService.Settings(model=model)
    )
    sent = await _one_turn(llm)
    assert sent["reasoning"] == {"effort": "medium"}


async def test_a_model_that_does_not_reason_gets_no_reasoning():
    llm = DograhOpenAIResponsesLLMService(
        api_key="k",
        settings=DograhOpenAIResponsesLLMService.Settings(model="gpt-4.1"),
    )
    assert "reasoning" not in await _one_turn(llm)


# --- node-transition deferral, ported from the chat service ------------------


def _text(delta):
    return ResponseTextDeltaEvent.model_construct(
        type="response.output_text.delta", delta=delta, item_id="msg_1"
    )


def _call(name, call_id):
    item = ResponseFunctionToolCall.model_construct(
        type="function_call",
        id=f"fc_{call_id}",
        name=name,
        call_id=call_id,
        arguments="{}",
    )
    return [
        ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=item
        ),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done", item=item
        ),
    ]


def _service():
    llm = DograhOpenAIResponsesLLMService(
        api_key="k",
        settings=DograhOpenAIResponsesLLMService.Settings(model="gpt-6-luna"),
    )
    llm.register_function("transition", AsyncMock(), is_node_transition=True)
    llm.register_function("lookup", AsyncMock())
    return llm


def _names(run):
    return [[fc.function_name for fc in call.args[0]] for call in run.await_args_list]


async def test_transition_during_a_spoken_response_waits_for_it_to_end():
    llm = _service()
    with patch.object(LLMService, "run_function_calls", new_callable=AsyncMock) as run:
        await _one_turn(llm, [_text("Le paso con ventas."), *_call("transition", "1")])
        run.assert_not_awaited()
        await llm.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        assert _names(run) == [["transition"]]


async def test_transition_without_speech_and_other_tools_run_at_once():
    llm = _service()
    with patch.object(LLMService, "run_function_calls", new_callable=AsyncMock) as run:
        await _one_turn(llm, [_text(" ... "), *_call("transition", "1")])
        await _one_turn(llm, [_text("Un momento."), *_call("lookup", "2")])
        assert _names(run) == [["transition"], ["lookup"]]


async def test_a_new_response_discards_a_transition_never_released():
    llm = _service()
    with patch.object(LLMService, "run_function_calls", new_callable=AsyncMock) as run:
        await _one_turn(llm, [_text("Le paso."), *_call("transition", "1")])
        await _one_turn(llm, [_text("Perdone, repito.")])
        await llm.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        run.assert_not_awaited()
