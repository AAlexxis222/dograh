"""Provider ranges applied by the cascade resolver (clamp + warning, §2.2).

Sources: Deepgram Flux docs (R4 §3, spec V16): eot_threshold 0.5-1.0,
eager_eot_threshold 0.3-0.9 and ≤ eot_threshold, eot_timeout_ms 500-60000;
ElevenLabs websocket speed 0.7-1.2 (pass2/service-factory.md B9). LLM
temperature 0-2 is the widest range any provider accepts. ElevenLabs realtime
STT documents its VAD ranges on the field (elevenlabs/stt.py:196-197):
vad_threshold 0.1-0.9, vad_silence_threshold_secs 0.3-3.0. Ultravox call
creation takes temperature 0-1 and maxDuration 10-3600 s; the row takes
max_duration as a number of seconds only, so every value passes this clamp.
"""

from __future__ import annotations

from typing import Any

_FLUX_PROVIDERS = ("deepgram", "dograh")

LLM_PROVIDERS = (
    "_all",
    "atlascloud",
    "aws_bedrock",
    "azure",
    "dograh",
    "google",
    "google_vertex",
    "groq",
    "huggingface",
    "minimax",
    "openai",
    "openrouter",
    "sarvam",
    "speaches",
)
"""Every ``llm`` key in ``service_tuning_specs.SPECS``, repeated here because
this module must stay importable without pipecat. Kept in step by
test_service_tuning_schema.py::test_llm_temperature_bounds_track_every_llm_row."""

SERVICE_TUNING_NUMERIC_BOUNDS: tuple[tuple[tuple[str, ...], float, float], ...] = tuple(
    [
        (("service_tuning", "stt", p, "settings", "eot_threshold"), 0.5, 1.0)
        for p in _FLUX_PROVIDERS
    ]
    + [
        (("service_tuning", "stt", p, "settings", "eager_eot_threshold"), 0.3, 0.9)
        for p in _FLUX_PROVIDERS
    ]
    + [
        (("service_tuning", "stt", p, "settings", "eot_timeout_ms"), 500, 60000)
        for p in _FLUX_PROVIDERS
    ]
    + [
        (("service_tuning", "tts", "elevenlabs", "settings", "speed"), 0.7, 1.2),
        (("service_tuning", "tts", "elevenlabs", "settings", "stability"), 0.0, 1.0),
        (
            ("service_tuning", "tts", "elevenlabs", "settings", "similarity_boost"),
            0.0,
            1.0,
        ),
        (("service_tuning", "tts", "elevenlabs", "settings", "style"), 0.0, 1.0),
    ]
    # ElevenLabs realtime STT VAD (elevenlabs/stt.py:196-197).
    + [
        (
            ("service_tuning", "stt", "elevenlabs", "settings", "vad_threshold"),
            0.1,
            0.9,
        ),
        (
            (
                "service_tuning",
                "stt",
                "elevenlabs",
                "settings",
                "vad_silence_threshold_secs",
            ),
            0.3,
            3.0,
        ),
    ]
    # Ultravox call creation (realtime): max_duration in seconds.
    + [
        (
            (
                "service_tuning",
                "realtime",
                "ultravox_realtime",
                "settings",
                "temperature",
            ),
            0.0,
            1.0,
        ),
        (
            (
                "service_tuning",
                "realtime",
                "ultravox_realtime",
                "settings",
                "max_duration",
            ),
            10,
            3600,
        ),
    ]
    + [
        (("service_tuning", "llm", p, "settings", "temperature"), 0.0, 2.0)
        for p in LLM_PROVIDERS
    ]
)


def _is_number(value: Any) -> bool:
    """``isinstance(True, int)`` is True, so a bool would compare as 1/0 and
    produce a nonsense warning ("True lowered to eot_threshold 0.7"). Same
    guard the clamp uses (cascade.py:196); the PUT rejects these anyway, and
    this module also runs over documents stored before it did."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def apply_service_tuning_invariants(effective: dict[str, Any]) -> list[str]:
    """Cross-field rules that a per-key clamp cannot express. Mutates in place."""
    warnings: list[str] = []
    stt = (effective.get("service_tuning") or {}).get("stt") or {}
    for provider in _FLUX_PROVIDERS:
        settings = (stt.get(provider) or {}).get("settings") or {}
        eager, eot = settings.get("eager_eot_threshold"), settings.get("eot_threshold")
        if _is_number(eager) and _is_number(eot) and eager > eot:
            settings["eager_eot_threshold"] = eot
            warnings.append(
                f"service_tuning.stt.{provider}.settings.eager_eot_threshold: {eager} lowered to eot_threshold {eot}"
            )
    return warnings
