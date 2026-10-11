"""Reasoning effort matrix for OpenAI's own models (VOZ-AC-B1-47, VOZ-BUG-03).

Every impact knob the conversation LLM sends is declared here per (model,
api), so neither pipecat nor the API picks it by default (VOZ-AC-B1-17):
pipecat's Responses service sends ``effort: "none"`` when nobody sets it.

Values are OpenAI's documented ones (docs-closure-a PB-03, UV-51a-e). On
chat/completions the row reproduces what production sent before this table
existed ("exponer ≠ cambiar", VOZ-AC-B0-14): the gpt-5 family keeps
``reasoning_effort: minimal`` and ``verbosity: low``; models whose effort was
an implicit API default get the documented default (``medium``) declared.
Presets, not this table, change production defaults (VOZ-D13).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from typing import Any, Literal

Api = Literal["chat_completions", "responses"]
# When a knob may travel: always, only next to ``effort: "none"``, or never.
Rule = Literal["any", "only_with_effort_none", "never"]

_BOTH_APIS: frozenset[str] = frozenset({"chat_completions", "responses"})
_GPT5_TODAY = {"reasoning_effort": "minimal", "verbosity": "low"}
_DECLARED_MEDIUM = {"reasoning_effort": "medium"}
_EFFORT = "llm.openai.options.reasoning.effort"


class EffortError(ValueError):
    """A reasoning knob the selected model or API cannot take (B0-28 shape)."""

    def __init__(self, *, code: str, where: str, reason: str, hint: str):
        super().__init__(f"{where}: {reason}; {hint}")
        self.code = code
        self.where = where
        self.reason = reason
        self.hint = hint

    def as_detail(self) -> dict[str, str]:
        return {
            "code": self.code,
            "where": self.where,
            "reason": self.reason,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class EffortRow:
    allowed: frozenset[str]  # empty: the model does not reason
    default: str | None  # declared on Responses when nobody asks
    chat_default: Mapping[str, str]  # today's chat/completions extras
    temperature: Rule
    temperature_value: float | None  # today's value for this model
    tools_by_chat: Rule
    apis: frozenset[str] = field(default=_BOTH_APIS)


_NO_REASONING = EffortRow(
    allowed=frozenset(),
    default=None,
    chat_default={},
    temperature="any",
    temperature_value=0.1,
    tools_by_chat="any",
)
# Not an OpenAI model this table knows (o-series, OpenRouter catalogue, Azure
# deployment names): chat/completions keeps today's payload, Responses is
# refused rather than guessed.
_UNLISTED = EffortRow(
    allowed=frozenset(),
    default=None,
    chat_default={},
    temperature="any",
    temperature_value=0.1,
    tools_by_chat="any",
    apis=frozenset({"chat_completions"}),
)
_GPT5_ORIGINAL = EffortRow(
    allowed=frozenset({"minimal", "low", "medium", "high"}),
    default="medium",
    chat_default=_GPT5_TODAY,
    temperature="never",
    temperature_value=None,
    tools_by_chat="any",
)
# Today's chat/completions rule, kept for the names the globs below cannot
# read: a fine-tune (``ft:gpt-5-mini:org::id``) or an Azure deployment
# (``prod-gpt-5``). Responses is refused rather than guessed.
_GPT5_BY_NAME = replace(_GPT5_ORIGINAL, apis=frozenset({"chat_completions"}))
_GPT5_POINT = EffortRow(
    allowed=frozenset({"none", "low", "medium", "high", "xhigh"}),
    default="medium",
    chat_default=_GPT5_TODAY,
    temperature="only_with_effort_none",
    temperature_value=None,
    tools_by_chat="any",
)
_GPT5_6 = EffortRow(
    allowed=frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
    default="medium",
    chat_default=_GPT5_TODAY,
    temperature="only_with_effort_none",
    temperature_value=None,
    tools_by_chat="any",
)
_GPT6 = EffortRow(
    allowed=frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
    default="medium",
    chat_default=_DECLARED_MEDIUM,
    temperature="only_with_effort_none",
    temperature_value=0.1,
    tools_by_chat="only_with_effort_none",
)
# ``none`` is an HTTP 400 on these two, so temperature can never travel.
_GPT6_REASONING_ONLY = EffortRow(
    allowed=frozenset({"low", "medium", "high", "xhigh", "max"}),
    default="medium",
    chat_default=_DECLARED_MEDIUM,
    temperature="only_with_effort_none",
    temperature_value=0.1,
    tools_by_chat="never",
)

# Rows whose chat/completions effort follows the docs, not a pre-matrix
# payload: no explicit effort reached them before, so their range binds there.
_DOC_RANGE_ON_CHAT = (_GPT6, _GPT6_REASONING_ONLY)

# First match wins, so the narrow globs come before the family ones.
_ROWS: tuple[tuple[tuple[str, ...], EffortRow], ...] = (
    (("gpt-5-chat*", "gpt-4*", "gpt-3.5*"), _NO_REASONING),
    (("gpt-6-astra*", "gpt-6.1-sol*"), _GPT6_REASONING_ONLY),
    (("gpt-6-luna*", "gpt-6-sol*"), _GPT6),
    (("gpt-5.6*",), _GPT5_6),
    (("gpt-5.[1-5]*",), _GPT5_POINT),
    (("gpt-5", "gpt-5-*"), _GPT5_ORIGINAL),
)


def row_for(model: str) -> EffortRow:
    """The row for ``model``; an OpenRouter-style ``vendor/`` prefix is ignored.

    Case-sensitive, as today's ``"gpt-5" in model`` rule: an unread name that
    contains ``gpt-5`` keeps the gpt-5 row on chat/completions.
    """
    name = model.rsplit("/", 1)[-1]
    for patterns, row in _ROWS:
        if any(fnmatchcase(name, pattern) for pattern in patterns):
            return row
    if "gpt-5" in model and "gpt-5-chat" not in model:
        return _GPT5_BY_NAME
    return _UNLISTED


def reasons(model: str) -> bool:
    return bool(row_for(model).allowed)


def wire_params(
    *,
    model: str,
    requested_effort: str | None,
    api: Api = "responses",
    requested_verbosity: str | None = None,
    requested_summary: str | None = None,
) -> dict[str, Any]:
    """The reasoning, verbosity and temperature ``model`` sends over ``api``.

    Chat/completions keys are the request's own (``reasoning_effort``,
    ``verbosity``); Responses ones are nested (``reasoning``, ``text``).
    ``temperature`` is today's value for the model and travels only where
    the row's rule allows it with the effective effort.
    """
    row = row_for(model)
    if api not in row.apis:
        raise EffortError(
            code="api_not_supported",
            where="llm.openai.options.api",
            reason=f"{model} is not in the OpenAI effort matrix, so {api} "
            "cannot declare its reasoning",
            hint="use chat_completions or a model listed in openai_efforts",
        )
    # The gpt-5 family sends an explicit effort verbatim over chat/completions,
    # as it did before this table; the documented ranges bind elsewhere.
    if (
        (api == "responses" or row in _DOC_RANGE_ON_CHAT)
        and requested_effort is not None
        and requested_effort not in row.allowed
    ):
        raise EffortError(
            code="knob_out_of_range",
            where=_EFFORT,
            reason=f"{model} does not accept effort {requested_effort!r}; "
            f"it accepts {sorted(row.allowed) or 'no reasoning effort'}",
            hint=f"set {_EFFORT} to one of the accepted values or remove it",
        )
    if requested_verbosity is not None and not row.allowed:
        raise EffortError(
            code="knob_out_of_range",
            where="llm.openai.options.verbosity",
            reason=f"{model} does not reason and takes no verbosity",
            hint="remove llm.openai.options.verbosity or choose a gpt-5+ model",
        )
    if api == "chat_completions":
        if requested_summary is not None:
            raise EffortError(
                code="knob_out_of_range",
                where="llm.openai.options.reasoning.summary",
                reason="chat/completions carries no reasoning summary",
                hint="set llm.openai.options.api to responses or remove the summary",
            )
        params: dict[str, Any] = dict(row.chat_default)
        if requested_effort is not None:
            params["reasoning_effort"] = requested_effort
        if requested_verbosity is not None:
            params["verbosity"] = requested_verbosity
        effort = params.get("reasoning_effort")
    else:
        effort = requested_effort or row.default
        params = {}
        if effort is not None:
            params["reasoning"] = {"effort": effort}
            if requested_summary is not None:
                params["reasoning"]["summary"] = requested_summary
        if requested_verbosity is not None:
            params["text"] = {"verbosity": requested_verbosity}
    if row.temperature_value is not None and (
        row.temperature == "any"
        or (row.temperature == "only_with_effort_none" and effort == "none")
    ):
        params["temperature"] = row.temperature_value
    return params


def refuse_tools_by_chat(model: str, effort: str | None) -> None:
    """Raise where ``model`` takes no tools over chat/completions with ``effort``.

    The API answers a tool-carrying turn there with a 400 (UV-51d), so the
    run fails at start instead of on every turn.
    """
    rule = row_for(model).tools_by_chat
    if rule == "any" or (rule == "only_with_effort_none" and effort == "none"):
        return
    raise EffortError(
        code="knob_out_of_range",
        where="llm.openai.options.api",
        reason=f"{model} takes no tools over chat/completions"
        + (f" with effort {effort!r}" if rule == "only_with_effort_none" else ""),
        hint="set llm.openai.options.api to responses"
        + (f" or {_EFFORT} to none" if rule == "only_with_effort_none" else ""),
    )
