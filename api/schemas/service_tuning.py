"""``workflow_configurations.service_tuning`` — provider knobs as configuration.

Shape (spec §2.6 / §5): ``{stt|tts|llm|realtime: {<provider>|_all: {settings, ctor, options}}, scope}``.
``settings`` are fields of the provider's pipecat ``Settings`` (applied as a
delta via ``from_mapping``); ``ctor`` are allow-listed constructor kwargs;
``options`` are non-pipecat switches (``llm.openai.options.api``). Every key is
validated against ``api.services.pipecat.service_tuning_specs`` at PUT time.
Explicit ``null`` is kept (tri-state) and only accepted on nullable fields.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderTuning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: dict[str, Any] = Field(default_factory=dict)
    ctor: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class LLMScope(BaseModel):
    """Which secondary LLM instances the ``llm`` tuning also applies to."""

    model_config = ConfigDict(extra="forbid")

    inference: bool = False
    extraction: bool = False
    voicemail: bool = False
    filler: bool = False


class ServiceTuning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stt: dict[str, ProviderTuning] = Field(default_factory=dict)
    tts: dict[str, ProviderTuning] = Field(default_factory=dict)
    llm: dict[str, ProviderTuning] = Field(default_factory=dict)
    realtime: dict[str, ProviderTuning] = Field(default_factory=dict)
    scope: LLMScope = Field(default_factory=LLMScope)

    @model_validator(mode="after")
    def _known_knobs_only(self) -> "ServiceTuning":
        # Lazy import: the specs module imports pipecat service classes.
        from api.services.pipecat.service_tuning_specs import validate_service_tuning

        errors = validate_service_tuning(self.model_dump(exclude_unset=True))
        if errors:
            raise ValueError("; ".join(errors))
        return self
