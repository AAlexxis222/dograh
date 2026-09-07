"""Apply ``service_tuning`` on top of the factory's current defaults.

The factory keeps building each service exactly as today (base kwargs); the
plan is a delta merged over them and handed to ``Settings.from_mapping`` so
aliases resolve and explicit ``None`` survives (pipecat settings.py:275-315,
apply_update :258-264). Nothing is routed through ``extra``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeVar

from pipecat.services.settings import ServiceSettings

ALL = "_all"
LLMRole = Literal["conversation", "inference", "extraction", "voicemail", "filler"]
_S = TypeVar("_S", bound=ServiceSettings)


@dataclass(frozen=True)
class TuningPlan:
    settings: dict[str, Any]
    ctor: dict[str, Any]
    options: dict[str, Any]


EMPTY_PLAN = TuningPlan({}, {}, {})


def _section(document: dict | None, kind: str, provider: str) -> dict[str, Any]:
    return ((document or {}).get(kind) or {}).get(provider) or {}


def tuning_for(document: dict | None, kind: str, provider: str) -> TuningPlan:
    """Merge the kind's ``_all`` section under the provider's own, key by key.

    ``_all`` is read for every kind, but only ``tts`` and ``llm`` declare one
    in ``service_tuning_specs.SPECS``: ``stt._all`` and ``realtime._all`` are
    "unknown provider" at the PUT, so the merge below is a no-op for them
    until a row gives them a meaning.
    """
    common, own = _section(document, kind, ALL), _section(document, kind, provider)
    if not common and not own:
        return EMPTY_PLAN
    return TuningPlan(
        settings={**(common.get("settings") or {}), **(own.get("settings") or {})},
        ctor={**(common.get("ctor") or {}), **(own.get("ctor") or {})},
        options={**(common.get("options") or {}), **(own.get("options") or {})},
    )


def build_settings(
    settings_cls: type[_S], base: dict[str, Any], plan: TuningPlan
) -> _S:
    if not plan.settings:
        return settings_cls(**base)
    return settings_cls.from_mapping({**base, **plan.settings})


def coerce_settings(
    plan: TuningPlan, coercions: Mapping[str, Callable[[Any], Any]]
) -> TuningPlan:
    """Return a copy of ``plan`` with the named settings converted.

    A tuning document carries JSON, so a field declared as an enum or as a
    provider config object arrives as a string or a dict and
    ``from_mapping`` stores it unchanged — the service then reads
    ``.value`` (speechmatics/stt.py:752) or hands the dict to its SDK
    (speechmatics/stt.py:775). The branch converts it here instead. The
    converters must accept an already-converted value, and ``None`` (an
    explicit "unset this field") is left alone. Never mutates ``plan``.
    """
    if not plan.settings:
        return plan
    settings = dict(plan.settings)
    for name, convert in coercions.items():
        if settings.get(name) is not None:
            settings[name] = convert(settings[name])
    return replace(plan, settings=settings)


def llm_tuning_applies(document: dict | None, role: LLMRole) -> bool:
    if not document or not document.get("llm"):
        return False
    if role == "conversation":
        return True
    return bool((document.get("scope") or {}).get(role))


def llm_document_for_role(document: dict | None, role: LLMRole) -> dict | None:
    return document if llm_tuning_applies(document, role) else None
