from api.services.configuration import registry
from api.services.configuration.options.cartesia import CARTESIA_INK_2_STT_LANGUAGES


def _default(cls_name: str, field: str):
    return getattr(registry, cls_name).model_fields[field].default


def test_azure_speech_region_defaults_to_westeurope():
    regions = [c.model_fields["region"].default for c in vars(registry).values()
               if isinstance(c, type) and hasattr(c, "model_fields") and "region" in getattr(c, "model_fields", {})
               and "azure" in c.__name__.lower()]
    assert regions and all(r == "westeurope" for r in regions)


def test_ink2_lists_spanish():
    assert "es" in CARTESIA_INK_2_STT_LANGUAGES


# Explicit exemptions: no Spanish support documented in this codebase's options.
ENGLISH_DEFAULT_EXEMPT = {
    "SpeachesSTTConfiguration",  # default model is the English-only distil-whisper .en
}


def test_tts_defaults_are_not_english_only():
    english = [n for n, c in vars(registry).items() if isinstance(c, type) and hasattr(c, "model_fields")
               and "language" in getattr(c, "model_fields", {}) and str(c.model_fields["language"].default).startswith("en")
               and n not in ENGLISH_DEFAULT_EXEMPT]
    assert not english, f"TTS/STT defaults still English: {english}"


def test_task_literal_voice_and_model_defaults():
    assert _default("DeepgramTTSConfiguration", "voice") == "aura-2-carina-es"
    assert _default("GoogleTTSConfiguration", "voice") == "es-ES-Chirp3-HD-Kore"
    assert _default("CartesiaTTSConfiguration", "model") == "sonic-3.6"
    assert (
        _default("CartesiaTTSConfiguration", "voice")
        == "3faa81ae-d3d8-4ab1-9e44-e50e46d33c30"
    )
    assert _default("AzureSpeechTTSConfiguration", "voice") == "es-ES-ElviraNeural"


def test_new_defaults_are_listed_in_their_option_lists():
    from api.services.configuration.options.azure import AZURE_SPEECH_TTS_VOICES
    from api.services.configuration.options.google import GOOGLE_TTS_VOICES

    assert "es-ES" in registry.INWORLD_TTS_LANGUAGES
    assert _default("GoogleTTSConfiguration", "voice") in GOOGLE_TTS_VOICES
    assert _default("AzureSpeechTTSConfiguration", "voice") in AZURE_SPEECH_TTS_VOICES
