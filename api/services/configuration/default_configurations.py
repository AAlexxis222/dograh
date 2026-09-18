"""The "default configurations" envelope served to configuration editors.

Two routes build it: the anonymous platform defaults
(``GET /user/configurations/defaults``) and the per-organization effective
defaults (``GET /organizations/workflow-configuration-effective-defaults``).
Only ``workflow_configurations`` differs between them, so the rest of the
payload — provider schemas, default providers, disposition suggestions,
widget texts — is built here once.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from api.schemas.widget_texts import WidgetTexts
from api.schemas.workflow_configurations import (
    CallDispositionOption,
    TextChatInactivityTimeoutConstraints,
    WorkflowConfigurationDefaults,
    get_default_call_disposition_options,
)
from api.services.configuration.defaults import DEFAULT_SERVICE_PROVIDERS
from api.services.configuration.registry import REGISTRY, ServiceType


class DefaultConfigurationsResponse(BaseModel):
    llm: dict[str, dict]
    tts: dict[str, dict]
    stt: dict[str, dict]
    embeddings: dict[str, dict]
    realtime: dict[str, dict]
    default_providers: dict[str, str]
    workflow_configurations: WorkflowConfigurationDefaults
    default_call_dispositions: list[CallDispositionOption] = Field(
        description=(
            "Built-in suggestions for call-disposition extraction. They do not "
            "enable extraction until saved in workflow_configurations.call_dispositions."
        )
    )
    text_chat_inactivity_timeout_constraints: TextChatInactivityTimeoutConstraints
    widget_text_defaults: WidgetTexts


class EffectiveDefaultConfigurationsResponse(DefaultConfigurationsResponse):
    """The same envelope plus the cascade warnings raised while resolving the
    organization layer (clamped bounds, dropped list merges)."""

    warnings: list[str] = Field(default_factory=list)


def build_default_configurations_response(
    workflow_configurations: WorkflowConfigurationDefaults,
) -> dict[str, Any]:
    """Payload shared by the anonymous platform defaults and the per-org
    effective defaults; only ``workflow_configurations`` differs."""
    return {
        "llm": {
            provider: model_cls.model_json_schema()
            for provider, model_cls in REGISTRY[ServiceType.LLM].items()
        },
        "tts": {
            provider: model_cls.model_json_schema()
            for provider, model_cls in REGISTRY[ServiceType.TTS].items()
        },
        "stt": {
            provider: model_cls.model_json_schema()
            for provider, model_cls in REGISTRY[ServiceType.STT].items()
        },
        "embeddings": {
            provider: model_cls.model_json_schema()
            for provider, model_cls in REGISTRY[ServiceType.EMBEDDINGS].items()
        },
        "realtime": {
            provider: model_cls.model_json_schema()
            for provider, model_cls in REGISTRY[ServiceType.REALTIME].items()
        },
        "default_providers": DEFAULT_SERVICE_PROVIDERS,
        "workflow_configurations": workflow_configurations,
        "default_call_dispositions": get_default_call_disposition_options(),
        "text_chat_inactivity_timeout_constraints": (
            TextChatInactivityTimeoutConstraints()
        ),
        "widget_text_defaults": WidgetTexts(),
    }
