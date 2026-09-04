from types import SimpleNamespace

from api.schemas.workflow_configurations import (
    MAX_CALL_DURATION_SECONDS,
    WorkflowConfigurationDefaults,
    schema_defaults_document,
)
from api.services.configuration.cascade import (
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
    org = {"call_dispositions": [{"code": "a", "description": "x" * 2500}]}
    wf = {
        "call_dispositions_extend_org": True,
        "call_dispositions": [{"code": "b", "description": "y" * 2000}],
    }
    resolved = _resolve(org, wf)
    assert resolved.effective["call_dispositions"] == wf["call_dispositions"]
    assert resolved.warnings and resolved.warnings[0].startswith("call_dispositions:")


def test_org_layer_never_overrides_pinned_definition():
    resolved = _resolve(
        {"dictionary": "org words", "max_user_idle_timeout": 20.0},
        {"dictionary": "mine"},
    )
    assert resolved.effective["dictionary"] == "mine"
    assert resolved.effective["max_user_idle_timeout"] == 20.0


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
