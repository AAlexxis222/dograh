# Deepgram serves the same APIs from several regional hosts, and the host is what
# decides where audio is processed - and therefore which jurisdiction the audio
# lands in. api.eu.deepgram.com keeps /v1/listen, /v2/listen and /v1/speak inside
# EU infrastructure, which is what a European deployment needs to avoid an
# undeclared transfer, so it is the default: an operator who leaves the field
# alone is EU-hosted, and one who wants another region has to choose it.
# Regional hosts accept the same API keys as the US one.
# https://developers.deepgram.com/reference/custom-endpoints
DEEPGRAM_DEFAULT_BASE_URL = "https://api.eu.deepgram.com"
# Where each regional host processes data, in the words the endpoint field uses.
_DEEPGRAM_REGIONS = {
    DEEPGRAM_DEFAULT_BASE_URL: "the EU",
    "https://api.deepgram.com": "the US",
    "https://api.au.deepgram.com": "Australia",
}
DEEPGRAM_BASE_URLS = tuple(_DEEPGRAM_REGIONS)


def deepgram_base_url_description(subject: str) -> str:
    """Describe a Deepgram ``base_url`` field from the hosts above."""
    others = " or ".join(
        f"{url} for {region}"
        for url, region in _DEEPGRAM_REGIONS.items()
        if url != DEEPGRAM_DEFAULT_BASE_URL
    )
    return (
        f"Deepgram API endpoint. This is what decides where {subject} is "
        f"processed: the default, {DEEPGRAM_DEFAULT_BASE_URL}, keeps "
        f"processing inside {_DEEPGRAM_REGIONS[DEEPGRAM_DEFAULT_BASE_URL]}; "
        f"use {others}. The same API key works on every regional endpoint."
    )


DEEPGRAM_FLUX_MODELS = ("flux-general-en", "flux-general-multi")
DEEPGRAM_FLUX_MULTILINGUAL_LANGUAGES = (
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "nl",
    "pt",
    "ru",
)
DEEPGRAM_FLUX_MULTILINGUAL_LANGUAGE_OPTIONS = (
    "multi",
    *DEEPGRAM_FLUX_MULTILINGUAL_LANGUAGES,
)
DEEPGRAM_STT_MODELS = ("nova-3-general", "nova-3-medical", *DEEPGRAM_FLUX_MODELS)
DEEPGRAM_LANGUAGES = (
    "multi",
    "ar",
    "ar-AE",
    "ar-SA",
    "ar-QA",
    "ar-KW",
    "ar-SY",
    "ar-LB",
    "ar-PS",
    "ar-JO",
    "ar-EG",
    "ar-SD",
    "ar-TD",
    "ar-MA",
    "ar-DZ",
    "ar-TN",
    "ar-IQ",
    "ar-IR",
    "be",
    "bn",
    "bs",
    "bg",
    "ca",
    "cs",
    "da",
    "da-DK",
    "de",
    "de-CH",
    "el",
    "en",
    "en-US",
    "en-AU",
    "en-GB",
    "en-IN",
    "en-NZ",
    "es",
    "es-419",
    "et",
    "fa",
    "fi",
    "fr",
    "fr-CA",
    "he",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "kn",
    "ko",
    "ko-KR",
    "lt",
    "lv",
    "mk",
    "mr",
    "ms",
    "nl",
    "nl-BE",
    "no",
    "pl",
    "pt",
    "pt-BR",
    "pt-PT",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "sv-SE",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh-CN",
    "zh-TW",
)
