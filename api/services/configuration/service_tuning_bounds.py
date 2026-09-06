"""Provider ranges applied by the cascade resolver (clamp + warning, §2.2).

Sources: Deepgram Flux docs (R4 §3, spec V16): eot_threshold 0.5-1.0,
eager_eot_threshold 0.3-0.9 and ≤ eot_threshold, eot_timeout_ms 500-60000;
ElevenLabs websocket speed 0.7-1.2 (pass2/service-factory.md B9). LLM
temperature 0-2 is the widest range any provider accepts.
"""

from __future__ import annotations

from typing import Any

_FLUX_PROVIDERS = ("deepgram", "dograh")

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
        (("service_tuning", "llm", "_all", "settings", "temperature"), 0.0, 2.0),
    ]
)


def apply_service_tuning_invariants(effective: dict[str, Any]) -> list[str]:
    """Cross-field rules that a per-key clamp cannot express. Mutates in place."""
    warnings: list[str] = []
    stt = (effective.get("service_tuning") or {}).get("stt") or {}
    for provider in _FLUX_PROVIDERS:
        settings = (stt.get(provider) or {}).get("settings") or {}
        eager, eot = settings.get("eager_eot_threshold"), settings.get("eot_threshold")
        if (
            isinstance(eager, (int, float))
            and isinstance(eot, (int, float))
            and eager > eot
        ):
            settings["eager_eot_threshold"] = eot
            warnings.append(
                f"service_tuning.stt.{provider}.settings.eager_eot_threshold: {eager} lowered to eot_threshold {eot}"
            )
    return warnings
