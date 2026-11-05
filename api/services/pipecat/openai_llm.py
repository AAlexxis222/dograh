"""Every build decision for an LLM that serves OpenAI's own models.

The effort matrix (``openai_efforts``) says what each model takes; this
module turns a ``service_tuning`` plan into the service that honours it, or
refuses the plan at run start (spec §5.4, VOZ-AC-B1-47). The factory calls
it once per build: ``build_openai_llm`` for the OpenAI and Atlas Cloud wires,
``openai_chat_inputs`` for the wires that only speak chat/completions to
OpenAI models (OpenRouter, Azure, Dograh).

Every refusal is a ``KnobRefusal`` (B0-28 shape); ``_HTTP_STATUS`` is the one
place that maps its code to the run-start status.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import wraps
from typing import Any, TypeVar

from fastapi import HTTPException

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat import service_tuning_specs
from api.services.pipecat.openai_efforts import (
    KnobRefusal,
    reasons,
    refuse_tools_by_chat,
    row_for,
    wire_params,
)
from api.services.pipecat.openai_responses import DograhOpenAIResponsesLLMService
from api.services.pipecat.service_tuning import (
    EMPTY_PLAN,
    TuningPlan,
    build_settings,
    warn_dropped,
)
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesReasoningConfig

# Wires that speak chat/completions to OpenAI's own models through an
# ``OpenAILLMService`` subclass but build their own service (#12). Groq,
# HuggingFace, Speaches, MiniMax and Sarvam serve their own catalogues, which
# the matrix does not list.
OPENAI_CHAT_WIRES = frozenset(
    {
        ServiceProviders.OPENROUTER.value,
        ServiceProviders.AZURE.value,
        ServiceProviders.DOGRAH.value,
    }
)
# Knob refusals keep the 400 the factory always used at run start; a build
# gate is the 422 VOZ-AC-B1-18 names.
_HTTP_STATUS = {
    "knob_out_of_range": 400,
    "knob_not_supported": 400,
    "api_not_supported": 400,
    "build_gate": 422,
}
_API_LABELS = {"chat_completions": "chat/completions", "responses": "the Responses API"}
# Declared on ``OpenAILLMSettings`` but never read by the Responses service
# (responses/llm.py:378-390 sends only temperature, top_p and the cap).
_RESPONSES_UNREAD = frozenset({"frequency_penalty", "presence_penalty", "seed"})

_T = TypeVar("_T")


def _http_refusals(build: Callable[..., _T]) -> Callable[..., _T]:
    @wraps(build)
    def wrapper(*args, **kwargs) -> _T:
        try:
            return build(*args, **kwargs)
        except KnobRefusal as e:
            raise HTTPException(
                status_code=_HTTP_STATUS[e.code], detail=e.as_detail()
            ) from e

    return wrapper


def _api(plan: TuningPlan) -> str:
    """The api the plan asks for; ``responses`` behind its build gate
    (VOZ-AC-B1-18), read at build so turning the constant off stops stored
    workflows too, not only new PUTs."""
    api = plan.options.get("api", "chat_completions")
    if api == "responses" and not service_tuning_specs.RESPONSES_API_AVAILABLE:
        raise KnobRefusal(
            code="build_gate",
            where="llm.openai.options.api",
            reason="the Responses API is turned off in this build",
            hint="set llm.openai.options.api to chat_completions, or turn "
            "RESPONSES_API_AVAILABLE back on (VOZ-N0-25)",
            gate="RESPONSES_API_AVAILABLE",
            source="api/services/pipecat/service_tuning_specs.py "
            "RESPONSES_API_AVAILABLE",
        )
    return api


def _wire(model: str, api: str, options: Mapping[str, Any]) -> dict[str, Any]:
    reasoning = options.get("reasoning") or {}
    return wire_params(
        model=model,
        api=api,
        requested_effort=reasoning.get("effort"),
        requested_verbosity=options.get("verbosity"),
        requested_summary=reasoning.get("summary"),
    )


def _reject_unread_on_responses(plan: TuningPlan) -> None:
    unread = sorted(_RESPONSES_UNREAD & set(plan.settings))
    if unread:
        raise KnobRefusal(
            code="knob_not_supported",
            where=f"llm.openai.settings.{unread[0]}",
            reason="the Responses API does not read " + ", ".join(unread),
            hint="remove it or set llm.openai.options.api to chat_completions",
        )


def _reject_top_p_on_responses(plan: TuningPlan, model: str, wire: dict) -> None:
    """``top_p`` travels over Responses only where the row's rule lets it
    (UV-51d); anything else 400s every turn."""
    if "top_p" not in plan.settings:
        return
    rule = row_for(model).top_p
    effort = wire.get("reasoning", {}).get("effort")
    if rule == "any" or (rule == "only_with_effort_none" and effort == "none"):
        return
    raise KnobRefusal(
        code="knob_out_of_range",
        where="llm.openai.settings.top_p",
        reason=f"{model} takes top_p over the Responses API only with effort "
        f"none, and the effort is {effort!r}"
        if rule == "only_with_effort_none"
        else f"{model} takes no top_p over the Responses API",
        hint="remove llm.openai.settings.top_p"
        + (
            " or set llm.openai.options.reasoning.effort to none"
            if rule == "only_with_effort_none"
            else ""
        ),
    )


def _refuse_temperature(model: str, api: str) -> None:
    reason = f"llm temperature is not accepted by {model} over {_API_LABELS[api]}"
    if api == "chat_completions":
        # Today's text and status, pinned by the chat tests ("exponer ≠
        # cambiar"); it predates the B0-28 shape.
        raise HTTPException(
            status_code=400,
            detail=f"{reason}; remove llm.*.settings.temperature or choose "
            "another model",
        )
    raise KnobRefusal(
        code="knob_out_of_range",
        where="llm.openai.settings.temperature",
        reason=f"{reason} with an effort other than none",
        hint="remove llm.openai.settings.temperature; it travels only with "
        "llm.openai.options.reasoning.effort none, where the model accepts none",
    )


def _translate(plan: TuningPlan, model: str, api: str, wire: dict) -> TuningPlan:
    """Translate the plan into what ``model`` accepts over ``api``; a no-op
    for a model that does not reason over chat/completions.

    ``build_chat_completion_params`` sends ``temperature`` and ``max_tokens``
    verbatim (openai/base_llm.py:356-358), and a reasoning model rejects both
    over this API: the completion cap has to arrive as
    ``max_completion_tokens``, and temperature only travels where the matrix
    puts it in ``wire``. The Responses service reads the cap only as
    ``max_completion_tokens`` too. The cap is translated (lossless); a
    temperature the model cannot take fails the run start by name rather than
    starting a run that silently differs from the document (spec §1.2, R20).
    An explicitly tuned ``max_completion_tokens`` wins over a translated
    ``max_tokens``. Returns a copy; the plan is never mutated.
    """
    if api == "responses":
        _reject_unread_on_responses(plan)
        _reject_top_p_on_responses(plan, model, wire)
    elif not reasons(model):
        return plan
    if not {"temperature", "max_tokens"} & set(plan.settings):
        return plan
    if "temperature" in plan.settings and reasons(model) and "temperature" not in wire:
        _refuse_temperature(model, api)
    settings = {
        name: value for name, value in plan.settings.items() if name != "max_tokens"
    }
    if "max_tokens" in plan.settings:
        settings.setdefault("max_completion_tokens", plan.settings["max_tokens"])
    return replace(plan, settings=settings)


def _chat_base(model: str) -> dict[str, Any]:
    """Base settings for a chat/completions wire that defaults temperature:
    today's ``temperature: 0.1`` where the matrix lets it travel; a reasoning
    model rejects the field with any effort but ``none``, so the default
    would start the run and fail every turn (§1.2)."""
    wire = wire_params(model=model, requested_effort=None, api="chat_completions")
    return {"model": model, **_temperature_of(wire)}


def _temperature_of(wire: Mapping[str, Any]) -> dict[str, Any]:
    return {"temperature": wire["temperature"]} if "temperature" in wire else {}


@_http_refusals
def openai_chat_inputs(model: str, plan: TuningPlan) -> tuple[TuningPlan, dict]:
    """The translated plan and the base settings for an OpenAI-model wire
    that builds its own chat/completions service (``OPENAI_CHAT_WIRES``)."""
    api = _api(plan)
    plan = _translate(plan, model, api, _wire(model, api, plan.options))
    return plan, _chat_base(model)


def _responses_llm(api_key, model: str, wire: dict, plan: TuningPlan, kwargs: dict):
    """The Responses service with every matrix knob declared (VOZ-BUG-03)."""
    base: dict[str, Any] = {"model": model, **_temperature_of(wire)}
    if "reasoning" in wire:
        base["reasoning"] = OpenAIResponsesReasoningConfig(**wire["reasoning"])
    settings = build_settings(DograhOpenAIResponsesLLMService.Settings, base, plan)
    if "text" in wire:
        settings.extra["text"] = wire["text"]
    return DograhOpenAIResponsesLLMService(api_key=api_key, settings=settings, **kwargs)


@_http_refusals
def build_openai_llm(
    provider: str, model: str, api_key, plan: TuningPlan, kwargs: dict
):
    """The OpenAI/Atlas Cloud LLM, every impact knob declared from the effort
    matrix (spec §5.4, VOZ-AC-B1-47). The api and the matrix params are
    resolved once and shared by the plan translation and the build.
    ``extra`` is REGISTRY_OWNED, so the matrix keys are only ever reachable
    through ``options``, never through ``from_mapping``'s overflow."""
    api = _api(plan)
    options = plan.options
    if not reasons(model):
        dropped = {"reasoning", "verbosity"} & set(options)
        warn_dropped(
            "options",
            f"llm.{provider}",
            model,
            dropped,
            "only a reasoning model takes reasoning_effort/verbosity",
        )
        options = {k: v for k, v in options.items() if k not in dropped}
    wire = _wire(model, api, options)
    plan = _translate(plan, model, api, wire)
    if api == "responses":
        return _responses_llm(api_key, model, wire, plan, kwargs)
    # The service_tuning row reaches the LLM that carries the workflow's
    # tools; tool-less builds (voicemail classifier, extraction, QA) get no
    # row unless their scope opts in, so they are not refused here. Only
    # ``llm.openai`` declares the options the refusal's hint names, so the
    # other wires (Atlas Cloud included) keep today's per-turn behaviour.
    if provider == ServiceProviders.OPENAI.value and plan != EMPTY_PLAN:
        refuse_tools_by_chat(model, wire.get("reasoning_effort"))
    settings = build_settings(
        OpenAILLMSettings, {"model": model, **_temperature_of(wire)}, plan
    )
    settings.extra.update({k: v for k, v in wire.items() if k != "temperature"})
    return OpenAILLMService(api_key=api_key, settings=settings, **kwargs)
