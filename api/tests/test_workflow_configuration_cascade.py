from api.schemas.workflow_configurations import (
    WorkflowConfigurationDefaults,
    schema_defaults_document,
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
