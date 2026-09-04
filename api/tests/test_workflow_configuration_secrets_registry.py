from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.configuration.masking import mask_workflow_configurations
from api.services.configuration.secrets_registry import (
    SECRET_PATHS,
    find_secret_paths,
    mask_secrets,
)


def _secret_marked_paths(schema: dict, defs: dict, prefix=()) -> set[tuple[str, ...]]:
    found = set()
    for name, prop in schema.get("properties", {}).items():
        if prop.get("secret") is True:
            found.add(prefix + (name,))
        ref = prop.get("$ref") or next(
            (item.get("$ref") for item in prop.get("anyOf", []) if item.get("$ref")),
            None,
        )
        if ref:
            found |= _secret_marked_paths(
                defs[ref.rsplit("/", 1)[-1]], defs, prefix + (name,)
            )
    return found


def test_every_schema_field_marked_secret_is_registered():
    schema = WorkflowConfigurationDefaults.model_json_schema()
    marked = _secret_marked_paths(schema, schema.get("$defs", {}))
    assert marked == {("voicemail_detection", "api_key")}
    assert marked <= set(SECRET_PATHS)


def test_find_secret_paths_covers_model_overrides_v2_and_voicemail():
    document = {
        "max_call_duration": 300,
        "model_overrides": {"llm": {"provider": "openai", "api_key": "sk-llm"}},
        "model_configuration_v2_override": {"stt": {"deepgram": {"credentials": "c"}}},
        "voicemail_detection": {"enabled": True, "api_key": "sk-vm"},
    }
    assert sorted(find_secret_paths(document)) == [
        ("model_configuration_v2_override", "stt", "deepgram", "credentials"),
        ("model_overrides", "llm", "api_key"),
        ("voicemail_detection", "api_key"),
    ]
    assert find_secret_paths({"knowledge_base": {"api_key": "not-registered"}}) == []


def test_mask_secrets_masks_only_registered_paths_and_keeps_shape():
    document = {
        "model_overrides": {"llm": {"provider": "openai", "api_key": "sk-1234567890"}},
        "voicemail_detection": {"enabled": True, "api_key": "vm-1234567890"},
        "knowledge_base": {"api_key": "plain"},
    }
    masked = mask_secrets(document)
    assert masked["model_overrides"]["llm"]["api_key"].endswith("7890")
    assert "*" in masked["model_overrides"]["llm"]["api_key"]
    assert masked["voicemail_detection"]["api_key"].endswith("7890")
    assert masked["knowledge_base"]["api_key"] == "plain"
    assert document["voicemail_detection"]["api_key"] == "vm-1234567890"
    assert mask_secrets(None) is None and mask_secrets({}) == {}


def test_mask_workflow_configurations_delegates_to_registry():
    document = {"voicemail_detection": {"api_key": "vm-1234567890"}}
    assert mask_workflow_configurations(document) == mask_secrets(document)
