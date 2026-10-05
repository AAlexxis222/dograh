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
    "CambTTSConfiguration",  # no language options list for Camb in the repo
    "XAITTSConfiguration",  # no language options list for xAI in the repo
    "SpeachesSTTConfiguration",  # default model is the English-only distil-whisper .en
}


def test_tts_defaults_are_not_english_only():
    english = [n for n, c in vars(registry).items() if isinstance(c, type) and hasattr(c, "model_fields")
               and "language" in getattr(c, "model_fields", {}) and str(c.model_fields["language"].default).startswith("en")
               and n not in ENGLISH_DEFAULT_EXEMPT]
    assert not english, f"TTS/STT defaults still English: {english}"
