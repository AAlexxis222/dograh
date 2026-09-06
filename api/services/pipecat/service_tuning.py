"""Apply ``service_tuning`` on top of the factory's current defaults.

The factory keeps building each service exactly as today (base kwargs); the
plan is a delta merged over them and handed to ``Settings.from_mapping`` so
aliases resolve and explicit ``None`` survives (pipecat settings.py:275-315,
apply_update :258-264). Nothing is routed through ``extra``.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def llm_tuning_applies(document: dict | None, role: LLMRole) -> bool:
    if not document or not document.get("llm"):
        return False
    if role == "conversation":
        return True
    return bool((document.get("scope") or {}).get(role))


def llm_document_for_role(document: dict | None, role: LLMRole) -> dict | None:
    return document if llm_tuning_applies(document, role) else None
