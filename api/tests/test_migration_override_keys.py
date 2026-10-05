from api.services.configuration.override_keys import strip_secret_leaves

SECRET = "sk-test-canary-123"


def test_strip_removes_api_keys_from_both_override_shapes():
    doc = {
        "model_configuration_v2_override": {"llm": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": SECRET}},
        "model_overrides": {"tts": {"api_key": SECRET, "voice": "x"}},
    }
    out, removed = strip_secret_leaves(doc)
    assert SECRET not in str(out) and removed == 2
    assert out["model_overrides"]["tts"]["voice"] == "x"


def test_resolve_module_no_longer_copies_org_keys():
    import api.services.configuration.resolve as resolve
    assert not hasattr(resolve, "enrich_overrides_with_api_keys")
