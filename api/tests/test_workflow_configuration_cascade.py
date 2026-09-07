from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.schemas.workflow_configurations import (
    MAX_CALL_DURATION_SECONDS,
    WorkflowConfigurationDefaults,
    schema_defaults_document,
)
from api.services.configuration.ai_model_configuration import (
    migrate_workflow_configuration_model_override_to_v2,
)
from api.services.configuration.cascade import (
    ORGANIZATION_FORBIDDEN_KEYS,
    PASSTHROUGH_KEYS,
    load_effective_workflow_configurations,
    merge_configuration_documents,
    normalize_root_nulls,
    resolve_effective_workflow_configurations,
    run_configurations_for,
)
from api.services.configuration.default_configurations import (
    DefaultConfigurationsResponse,
    EffectiveDefaultConfigurationsResponse,
    build_default_configurations_response,
)

# A minimal, valid effective AI-model configuration. Inlined rather than
# imported from the pipeline integration helpers: this module is a fast unit
# suite and must not pull in the processor chain.
USER_CONFIGURATION = {
    "is_realtime": False,
    "stt": {"provider": "deepgram", "model": "nova-3", "api_key": "test-key"},
    "tts": {
        "provider": "cartesia",
        "model": "sonic-2",
        "api_key": "test-key",
        "voice_id": "test-voice",
    },
    "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "test-key"},
}


def test_schema_defaults_document_has_no_null_and_no_user_turn_stop_timeout():
    """The effective document is materialised from these defaults and the
    engine does ``float(run_configs["user_turn_stop_timeout"])`` when the key
    is present (run_pipeline.py:129), so nullable shadows must stay absent."""
    document = schema_defaults_document()
    assert None not in document.values()
    assert "user_turn_stop_timeout" not in document
    assert "model_overrides" not in document
    assert "model_configuration_v2_override" not in document
    assert document["voicemail_detection"] == {"enabled": False}
    assert document["transcript_configuration"] == {"include_end_timestamps": False}
    assert document["call_dispositions_extend_org"] is False
    assert document["max_call_duration"] == 300


def test_shadow_sections_keep_unknown_keys_and_mark_voicemail_api_key_secret():
    model = WorkflowConfigurationDefaults.model_validate(
        {
            "voicemail_detection": {
                "enabled": True,
                "provider": "openai",
                "api_key": "sk-1",
                "model": "x",
            }
        }
    )
    dumped = model.model_dump(exclude_unset=True)
    assert dumped == {
        "voicemail_detection": {
            "enabled": True,
            "provider": "openai",
            "api_key": "sk-1",
            "model": "x",
        }
    }
    schema = WorkflowConfigurationDefaults.model_json_schema()
    voicemail = schema["$defs"]["VoicemailDetectionConfiguration"]["properties"][
        "api_key"
    ]
    assert voicemail.get("secret") is True


def _resolve(org=None, wf=None):
    return resolve_effective_workflow_configurations(
        organization_defaults=org, definition_configurations=wf
    )


def test_empty_layers_resolve_to_schema_defaults():
    resolved = _resolve({}, {})
    assert resolved.effective == schema_defaults_document()
    assert resolved.warnings == []


def test_sparse_layers_three_deep_scalar_precedence():
    resolved = _resolve({"max_call_duration": 600}, {"max_call_duration": 900})
    assert resolved.effective["max_call_duration"] == 900
    assert (
        _resolve({"max_call_duration": 600}, {}).effective["max_call_duration"] == 600
    )


def test_nested_section_merges_key_by_key():
    resolved = _resolve(
        {"ambient_noise_configuration": {"enabled": True, "volume": 0.5}},
        {"ambient_noise_configuration": {"volume": 0.1}},
    )
    assert resolved.effective["ambient_noise_configuration"] == {
        "enabled": True,
        "volume": 0.1,
    }


def test_lists_replace_by_default():
    org = {"call_dispositions": [{"code": "a", "description": "A"}]}
    wf = {"call_dispositions": [{"code": "b", "description": "B"}]}
    assert _resolve(org, wf).effective["call_dispositions"] == wf["call_dispositions"]


def test_call_dispositions_extend_org_appends_unique_by_code():
    org = {
        "call_dispositions": [
            {"code": "a", "description": "A"},
            {"code": "b", "description": "B"},
        ]
    }
    wf = {
        "call_dispositions_extend_org": True,
        "call_dispositions": [
            {"code": "B", "description": "mine"},
            {"code": "c", "description": "C"},
        ],
    }
    assert _resolve(org, wf).effective["call_dispositions"] == [
        {"code": "a", "description": "A"},
        {
            "code": "B",
            "description": "mine",
        },  # workflow's own definition wins on the same code
        {"code": "c", "description": "C"},
    ]


def test_turn_ignore_terms_merge_defaults_appends_unique():
    """Rule declared by path now; the ``turn`` section itself lands with the
    turn-detection PR and reaches the merger through ``extra='allow'``."""
    org = {"turn": {"ignore_terms": {"terms": ["uh", "um"]}}}
    wf = {"turn": {"ignore_terms": {"merge_defaults": True, "terms": ["um", "hmm"]}}}
    assert _resolve(org, wf).effective["turn"]["ignore_terms"]["terms"] == [
        "uh",
        "um",
        "hmm",
    ]


def test_nested_null_is_preserved_and_root_null_is_normalised():
    resolved = _resolve(
        {"max_call_duration": None}, {"turn": {"analyzer": {"url": None}}}
    )
    assert resolved.effective["max_call_duration"] == 300
    assert resolved.effective["turn"]["analyzer"] == {"url": None}
    assert normalize_root_nulls({"a": None, "b": {"c": None}}) == {"b": {"c": None}}


def test_out_of_schema_keys_merge_with_the_same_rules():
    org = {"knowledge_base": {"top_k": 3, "timeout_ms": 500}}
    wf = {"knowledge_base": {"top_k": 5}}
    assert _resolve(org, wf).effective["knowledge_base"] == {
        "top_k": 5,
        "timeout_ms": 500,
    }


def test_passthrough_model_sections_replace_whole_and_lowest_layer_wins():
    org = {"model_overrides": {"llm": {"provider": "openai", "api_key": "org"}}}
    wf = {"model_overrides": {"tts": {"provider": "deepgram"}}}
    assert _resolve(org, wf).effective["model_overrides"] == wf["model_overrides"]
    assert _resolve(org, {}).effective["model_overrides"] == org["model_overrides"]


def test_numeric_bound_is_clamped_in_effective_with_warning():
    resolved = _resolve({}, {"max_call_duration": 5000})
    assert resolved.effective["max_call_duration"] == MAX_CALL_DURATION_SECONDS
    assert resolved.warnings == [
        f"max_call_duration: 5000 clamped to {MAX_CALL_DURATION_SECONDS} (source=workflow)"
    ]


def test_list_invariant_violation_falls_back_to_lower_layer_with_warning():
    """append-unique cannot duplicate codes, but the total description budget
    (MAX_CALL_DISPOSITION_DESCRIPTIONS_TOTAL_LENGTH = 4000) is a list invariant
    the union of two valid layers can exceed."""
    org = {
        "call_dispositions": [
            {"code": f"o{index}", "description": "x" * 1000} for index in range(3)
        ]
    }
    wf = {
        "call_dispositions_extend_org": True,
        "call_dispositions": [
            {"code": "w1", "description": "y" * 1000},
            {"code": "w2", "description": "z" * 1000},
        ],
    }
    resolved = _resolve(org, wf)
    assert resolved.effective["call_dispositions"] == wf["call_dispositions"]
    assert resolved.warnings and resolved.warnings[0].startswith("call_dispositions:")


def test_definition_layer_wins_over_the_organization_layer_key_by_key():
    resolved = _resolve(
        {"dictionary": "org words", "max_user_idle_timeout": 20.0},
        {"dictionary": "mine"},
    )
    assert resolved.effective["dictionary"] == "mine"
    assert resolved.effective["max_user_idle_timeout"] == 20.0


def test_organization_forbidden_keys_are_the_pbx_and_ai_model_sections():
    """The PUT's 422 list, pinned: the PBX keys are gated on the workflow save
    and the AI-model sections have their own organization level."""
    assert ORGANIZATION_FORBIDDEN_KEYS == {
        "external_pbx_field_mappings",
        "external_pbx_lead_headers",
        "model_overrides",
        "model_configuration_v2_override",
    }
    assert PASSTHROUGH_KEYS <= ORGANIZATION_FORBIDDEN_KEYS


def test_invalid_fallback_catalog_resolves_to_an_empty_one_with_two_warnings():
    """Nothing validates a document written straight to the store, so the layer
    the resolver falls back to can be invalid on its own."""
    org = {
        "call_dispositions": [
            {"code": f"o{index}", "description": "o" * 1000} for index in range(3)
        ]
    }
    wf = {
        "call_dispositions_extend_org": True,
        # The union blows the total description budget, and this layer on its
        # own repeats a code under casefold, so neither catalog validates.
        "call_dispositions": [
            {"code": "d1", "description": "a" * 1000},
            {"code": "D1", "description": "b" * 1000},
            {"code": "d2", "description": "c" * 1000},
        ],
    }
    resolved = _resolve(org, wf)
    assert resolved.effective["call_dispositions"] == []
    assert len(resolved.warnings) == 2
    assert resolved.warnings[1].endswith("resolved to an empty catalog")


def test_merge_does_not_mutate_inputs():
    base = {"ambient_noise_configuration": {"enabled": False}}
    overlay = {"ambient_noise_configuration": {"volume": 0.2}}
    merge_configuration_documents(base, overlay)
    assert base == {"ambient_noise_configuration": {"enabled": False}}
    assert overlay == {"ambient_noise_configuration": {"volume": 0.2}}


def test_run_configurations_for_prefers_frozen_and_falls_back_to_pinned():
    frozen = SimpleNamespace(
        effective_configurations={"a": 1},
        definition=SimpleNamespace(workflow_configurations={"a": 2}),
    )
    legacy = SimpleNamespace(
        effective_configurations=None,
        definition=SimpleNamespace(workflow_configurations={"a": 2}),
    )
    orphan = SimpleNamespace(effective_configurations=None, definition=None)
    assert run_configurations_for(frozen) == {"a": 1}
    assert run_configurations_for(legacy) == {"a": 2}
    assert run_configurations_for(orphan) == {}


@pytest.mark.asyncio
async def test_legacy_user_scoped_workflow_skips_the_tenant_check_and_org_layer():
    """``WorkflowModel.organization_id`` is nullable: such a workflow has no
    organization layer to inherit and no tenant to compare the definition to."""
    get_configuration_value = AsyncMock(return_value={"max_call_duration": 600})
    db = SimpleNamespace(
        get_definition_configurations_with_owner=AsyncMock(
            return_value=({"dictionary": "mine"}, 99)
        ),
        get_configuration_value=get_configuration_value,
    )

    resolved = await load_effective_workflow_configurations(
        db, organization_id=None, definition_id=11
    )

    assert resolved.effective["dictionary"] == "mine"
    assert resolved.effective["max_call_duration"] == 300
    get_configuration_value.assert_not_awaited()


def test_effective_envelope_carries_warnings_and_the_shared_payload():
    """The effective-defaults route serves the same envelope as the anonymous
    platform defaults, plus the warnings raised while resolving the org layer."""
    resolved = _resolve({"max_call_duration": MAX_CALL_DURATION_SECONDS + 1}, {})
    payload = build_default_configurations_response(
        WorkflowConfigurationDefaults.model_validate(resolved.effective)
    )
    envelope = EffectiveDefaultConfigurationsResponse(
        **payload, warnings=resolved.warnings
    )

    assert envelope.workflow_configurations.max_call_duration == (
        MAX_CALL_DURATION_SECONDS
    )
    assert envelope.warnings == resolved.warnings and envelope.warnings
    # Same component as the anonymous endpoint: nothing but warnings is added.
    assert set(EffectiveDefaultConfigurationsResponse.model_fields) - set(
        DefaultConfigurationsResponse.model_fields
    ) == {"warnings"}


def test_v2_migration_preserves_sections_it_does_not_own():
    base = EffectiveAIModelConfiguration.model_validate(USER_CONFIGURATION)
    document = {
        "model_overrides": {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "k"}
        },
        "service_tuning": {"tts": {"_all": {"silence_time_s": 0.4}}},
        "turn": {"source": "auto"},
        "soft_timeout": {"timeout_seconds": 4},
        "voicemail_detection": {"enabled": True},
    }
    migrated, changed = migrate_workflow_configuration_model_override_to_v2(
        document, base
    )
    assert changed is True
    assert "model_overrides" not in migrated
    for key in ("service_tuning", "turn", "soft_timeout", "voicemail_detection"):
        assert migrated[key] == document[key]


def test_workflow_effective_layers_are_consistent():
    from api.services.configuration.workflow_effective import (
        resolve_workflow_effective_configurations,
    )

    layers = resolve_workflow_effective_configurations(
        organization_defaults={"max_call_duration": 600},
        definition_configurations={"dictionary": "mine", "max_call_duration": None},
    )
    assert layers.own == {"dictionary": "mine"}  # root null normalised away
    assert layers.base["max_call_duration"] == 600
    assert layers.effective["max_call_duration"] == 600
    assert layers.effective["dictionary"] == "mine"
    assert layers.base["dictionary"] == ""  # schema default


def test_flux_thresholds_are_clamped_with_warning():
    resolved = resolve_effective_workflow_configurations(
        organization_defaults={
            "service_tuning": {
                "stt": {
                    "deepgram": {
                        "settings": {"eot_threshold": 1.4, "eot_timeout_ms": 100}
                    }
                }
            }
        },
        definition_configurations={},
    )
    s = resolved.effective["service_tuning"]["stt"]["deepgram"]["settings"]
    assert s["eot_threshold"] == 1.0 and s["eot_timeout_ms"] == 500
    assert any(
        w.startswith(
            "service_tuning.stt.deepgram.settings.eot_threshold: 1.4 clamped to 1.0"
        )
        for w in resolved.warnings
    )


def test_eager_above_eot_is_lowered_to_eot():
    resolved = resolve_effective_workflow_configurations(
        organization_defaults={
            "service_tuning": {"stt": {"dograh": {"settings": {"eot_threshold": 0.6}}}}
        },
        definition_configurations={
            "service_tuning": {
                "stt": {"dograh": {"settings": {"eager_eot_threshold": 0.8}}}
            }
        },
    )
    s = resolved.effective["service_tuning"]["stt"]["dograh"]["settings"]
    assert s["eager_eot_threshold"] == 0.6
    assert (
        "service_tuning.stt.dograh.settings.eager_eot_threshold: 0.8 lowered to eot_threshold 0.6"
        in resolved.warnings
    )


def test_eager_null_is_kept_and_not_clamped():
    resolved = resolve_effective_workflow_configurations(
        organization_defaults={
            "service_tuning": {
                "stt": {"deepgram": {"settings": {"eager_eot_threshold": 0.5}}}
            }
        },
        definition_configurations={
            "service_tuning": {
                "stt": {"deepgram": {"settings": {"eager_eot_threshold": None}}}
            }
        },
    )
    assert (
        resolved.effective["service_tuning"]["stt"]["deepgram"]["settings"][
            "eager_eot_threshold"
        ]
        is None
    )


def test_bool_thresholds_are_left_alone_by_the_cross_field_rule():
    # ``isinstance(True, int)`` is True, so the invariant used to compare a
    # bool as 1 and emit "True lowered to eot_threshold 0.7". The clamp
    # already skips bools (cascade.py:196) and so does this now: the PUT
    # rejects them, and a document stored before it did must not grow a
    # nonsense warning.
    resolved = resolve_effective_workflow_configurations(
        organization_defaults={},
        definition_configurations={
            "service_tuning": {
                "stt": {
                    "dograh": {
                        "settings": {"eager_eot_threshold": True, "eot_threshold": 0.7}
                    }
                }
            }
        },
    )
    s = resolved.effective["service_tuning"]["stt"]["dograh"]["settings"]
    assert s["eager_eot_threshold"] is True
    assert not [w for w in resolved.warnings if "eager_eot_threshold" in w]


def test_elevenlabs_ws_speed_clamped_to_supported_range():
    resolved = resolve_effective_workflow_configurations(
        organization_defaults={},
        definition_configurations={
            "service_tuning": {"tts": {"elevenlabs": {"settings": {"speed": 1.9}}}}
        },
    )
    assert (
        resolved.effective["service_tuning"]["tts"]["elevenlabs"]["settings"]["speed"]
        == 1.2
    )


@pytest.mark.parametrize(
    "kind,provider,name,written,used",
    [
        # Ultravox call-creation body: temperature 0-1, maxDuration in seconds
        # (ultravox/llm.py:348 renders the timedelta as "<seconds>s").
        ("realtime", "ultravox_realtime", "temperature", 1.7, 1.0),
        ("realtime", "ultravox_realtime", "max_duration", 5, 10),
        ("realtime", "ultravox_realtime", "max_duration", 7200, 3600),
        # elevenlabs/stt.py:196-197 documents both ranges on the field.
        ("stt", "elevenlabs", "vad_threshold", 0.95, 0.9),
        ("stt", "elevenlabs", "vad_threshold", 0.05, 0.1),
        ("stt", "elevenlabs", "vad_silence_threshold_secs", 0.1, 0.3),
        ("stt", "elevenlabs", "vad_silence_threshold_secs", 5.0, 3.0),
    ],
)
def test_ultravox_and_elevenlabs_vad_knobs_are_clamped(
    kind, provider, name, written, used
):
    resolved = resolve_effective_workflow_configurations(
        organization_defaults={},
        definition_configurations={
            "service_tuning": {kind: {provider: {"settings": {name: written}}}}
        },
    )
    assert (
        resolved.effective["service_tuning"][kind][provider]["settings"][name] == used
    )
    assert any(
        w.startswith(
            f"service_tuning.{kind}.{provider}.settings.{name}: {written} clamped to {used}"
        )
        for w in resolved.warnings
    )
