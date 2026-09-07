from __future__ import annotations

"""Helpers for merging incoming user-configuration updates with what is already
stored, while honouring masked API keys.
"""

import copy
from typing import Any, Dict

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration.masking import (
    MODEL_OVERRIDE_FIELDS,
    SERVICE_SECRET_FIELDS,
    VOICEMAIL_DETECTION_KEY,
    contains_masked_key,
    is_mask_of,
    resolve_masked_api_keys,
)
from api.services.configuration.secrets_registry import find_secret_paths

SERVICE_FIELDS = ("llm", "tts", "stt", "embeddings", "realtime")


def _same_provider(incoming_cfg: dict, existing_cfg: dict) -> bool:
    return not (
        existing_cfg.get("provider") is not None
        and incoming_cfg.get("provider") is not None
        and incoming_cfg.get("provider") != existing_cfg.get("provider")
    )


def _merge_service_secret_fields(
    incoming_cfg: dict,
    existing_cfg: dict,
    *,
    preserve_missing: bool,
    masked_value_preserves_full_secret: bool = False,
) -> dict:
    """Restore existing real secrets when incoming values are masked.

    If ``preserve_missing`` is true, missing incoming secret fields are also
    copied from the existing config. User config updates need that behavior;
    workflow model overrides leave missing secrets blank so later enrichment can
    copy from the current global config.
    """
    if not _same_provider(incoming_cfg, existing_cfg):
        return incoming_cfg

    for secret_field in SERVICE_SECRET_FIELDS:
        if secret_field not in existing_cfg:
            continue

        incoming_secret = incoming_cfg.get(secret_field)
        existing_secret = existing_cfg[secret_field]
        if incoming_secret is not None:
            if contains_masked_key(incoming_secret):
                incoming_cfg[secret_field] = (
                    existing_secret
                    if masked_value_preserves_full_secret
                    else resolve_masked_api_keys(
                        incoming_secret,
                        existing_secret,
                    )
                )
        elif preserve_missing:
            incoming_cfg[secret_field] = existing_secret

    return incoming_cfg


def merge_user_configurations(
    existing: EffectiveAIModelConfiguration, incoming_partial: Dict[str, dict]
) -> EffectiveAIModelConfiguration:
    """Merge *incoming_partial* onto *existing* and return a new EffectiveAIModelConfiguration.

    *incoming_partial* is the body of the PUT request (already `model_dump()`ed or
    extracted via Pydantic `model_dump`).

    Rules:
    1. If a service block is absent in the request, keep the existing one.
    2. If provider unchanged and the api_key field is either missing or equal to
       the masked placeholder, preserve the existing real key.
    3. If provider changes, the incoming api_key is used verbatim (validation
       will fail later if it is missing).
    4. Non-service top-level fields (e.g. `test_phone_number`) are overwritten
       when supplied.
    """

    merged = existing.model_dump(exclude_none=True)

    def _merge_service_block(service_name: str):
        incoming_cfg = incoming_partial.get(service_name)
        if incoming_cfg is None:
            return  # nothing to do

        old_cfg = merged.get(service_name, {})
        if old_cfg:
            incoming_cfg = _merge_service_secret_fields(
                incoming_cfg,
                old_cfg,
                preserve_missing=True,
            )

        merged[service_name] = incoming_cfg

    for service in SERVICE_FIELDS:
        _merge_service_block(service)

    # other simple fields
    if "is_realtime" in incoming_partial:
        merged["is_realtime"] = incoming_partial["is_realtime"]

    if "test_phone_number" in incoming_partial:
        merged["test_phone_number"] = incoming_partial["test_phone_number"]

    if "timezone" in incoming_partial:
        merged["timezone"] = incoming_partial["timezone"]

    return EffectiveAIModelConfiguration.model_validate(merged)


def _merge_voicemail_secret(merged: dict, existing_config: dict) -> None:
    """Restore the stored voicemail_detection.api_key in place.

    Responses mask the key, and the settings dialog re-sends the whole
    ``voicemail_detection`` object (sometimes without ``api_key`` at all when
    toggling ``use_workflow_llm``), so a masked or missing key means "keep the
    stored one"; only a new plain value replaces it. The stored key belongs to
    the stored provider: when the request names a different one, the mask is
    dropped instead of restored so the old key never authenticates against the
    new provider. Non-string values are left as they are (the block is
    ``extra="allow"``, so nothing upstream typed them).
    """
    incoming = merged.get(VOICEMAIL_DETECTION_KEY)
    existing = existing_config.get(VOICEMAIL_DETECTION_KEY)
    if not isinstance(incoming, dict) or not isinstance(existing, dict):
        return
    existing_key = existing.get("api_key")
    if not isinstance(existing_key, str) or not existing_key:
        return
    incoming_key = incoming.get("api_key")
    if incoming_key is not None and not (
        isinstance(incoming_key, str) and contains_masked_key(incoming_key)
    ):
        return
    incoming_provider = incoming.get("provider")
    if incoming_provider is not None and incoming_provider != existing.get("provider"):
        incoming.pop("api_key", None)
        return
    incoming["api_key"] = existing_key


# Root keys whose registered secrets are restored by their own dedicated
# merge above (voicemail_detection) or by an explicit masked-value check
# elsewhere (model_overrides via MODEL_OVERRIDE_FIELDS below;
# model_configuration_v2_override in ai_model_configuration.py — see the
# secrets_registry.py module docstring for why that one is deliberately not
# restored here).
_SECRET_ROOTS_MERGED_ELSEWHERE = frozenset(
    {VOICEMAIL_DETECTION_KEY, "model_overrides", "model_configuration_v2_override"}
)


def _merge_registered_secret_leaves(merged: dict, existing_config: dict) -> None:
    """Restore any other registered secret leaf (e.g. ``service_tuning.*.*.ctor.url``)
    when the incoming value is missing or the mask of the stored one."""
    for path in find_secret_paths(existing_config):
        if path[0] in _SECRET_ROOTS_MERGED_ELSEWHERE:
            continue
        existing_node: Any = existing_config
        incoming_node: Any = merged
        for part in path[:-1]:
            if not isinstance(existing_node, dict) or part not in existing_node:
                existing_node = None
                break
            existing_node = existing_node[part]
            if not isinstance(incoming_node, dict):
                incoming_node = None
                continue
            incoming_node = incoming_node.get(part)
        if not isinstance(existing_node, dict) or not isinstance(incoming_node, dict):
            continue
        leaf = path[-1]
        existing_value = existing_node.get(leaf)
        if not isinstance(existing_value, str) or not existing_value:
            continue
        incoming_value = incoming_node.get(leaf)
        if incoming_value is None or (
            isinstance(incoming_value, str)
            and is_mask_of(incoming_value, existing_value)
        ):
            incoming_node[leaf] = existing_value


def merge_workflow_configuration_secrets(
    incoming_config: dict | None,
    existing_config: dict | None,
) -> dict | None:
    """Restore persisted workflow override secrets when the client sends masks.

    Workflow model overrides intentionally persist real keys so a workflow keeps
    running after the global provider changes. API responses mask those keys, so
    save requests must merge masked placeholders back to the stored real values.

    Unlike user config updates, a missing workflow override secret is not copied
    from the existing workflow config. Missing means "copy from current global"
    during the later enrichment step.
    """
    if not incoming_config or not existing_config:
        return incoming_config

    merged = copy.deepcopy(incoming_config)
    _merge_voicemail_secret(merged, existing_config)
    _merge_registered_secret_leaves(merged, existing_config)

    incoming_overrides = merged.get("model_overrides")
    existing_overrides = existing_config.get("model_overrides")
    if not isinstance(incoming_overrides, dict) or not isinstance(
        existing_overrides, dict
    ):
        return merged

    for section in MODEL_OVERRIDE_FIELDS:
        incoming_section = incoming_overrides.get(section)
        existing_section = existing_overrides.get(section)
        if not isinstance(incoming_section, dict) or not isinstance(
            existing_section, dict
        ):
            continue

        incoming_overrides[section] = _merge_service_secret_fields(
            incoming_section,
            existing_section,
            preserve_missing=False,
            masked_value_preserves_full_secret=True,
        )

    return merged
