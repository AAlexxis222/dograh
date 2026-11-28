"""The URL a provider's client connects to, from the value the tenant stored.

Only the Deepgram and ElevenLabs clients normalise what they are given; every other provider hands the stored value
to its SDK as is. The service URL guard judges the exact string the client uses, so the factory and the
configuration validator both pass these functions' output to it (``service_client_url``).
"""

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse, urlunparse

from api.services.configuration.options import DEEPGRAM_DEFAULT_BASE_URL
from api.services.configuration.registry import ServiceProviders

# The fields of a service config section that hold a URL the tenant set.
SERVICE_URL_FIELDS = ("base_url", "endpoint")


def deepgram_client_url(base_url: str) -> str:
    """The Deepgram endpoint for a stored base_url: blank is the default host, a bare host gets https.

    Deepgram documents the regional switch as "replace api.deepgram.com with api.eu.deepgram.com", so operators
    reasonably type a bare host; TLS is assumed rather than rejecting a value that is obviously well intentioned.
    """
    base_url = base_url.strip()
    if not base_url:
        return DEEPGRAM_DEFAULT_BASE_URL
    if "://" not in base_url:
        base_url = f"https://{base_url}"
    return base_url.rstrip("/")


def elevenlabs_websocket_url(base_url: str) -> str:
    """Normalize an ElevenLabs API base URL for WebSocket clients."""
    base_url = base_url.strip()
    parsed = urlparse(base_url)
    if not parsed.netloc:
        return base_url.rstrip("/")

    websocket_scheme = {
        "http": "ws",
        "https": "wss",
    }.get(parsed.scheme, parsed.scheme)
    return urlunparse(
        parsed._replace(
            scheme=websocket_scheme,
            path=parsed.path.rstrip("/"),
        )
    )


def elevenlabs_realtime_stt_host(base_url: str) -> str:
    """Return the host/path prefix Pipecat's ElevenLabs realtime STT expects.

    Pipecat's realtime STT service builds
    ``wss://{host}/v1/speech-to-text/realtime`` internally, so remove the scheme
    from the same normalized WebSocket URL used by ElevenLabs TTS. Preserve
    netloc (including optional ports) and any path prefix used by BYOK proxies.
    """
    websocket_url = elevenlabs_websocket_url(base_url)
    parsed = urlparse(websocket_url)
    if parsed.netloc:
        path = parsed.path
        return f"{parsed.netloc}{path}" if path else parsed.netloc
    return websocket_url


def service_client_url(provider: str | None, role: str, url: str) -> str:
    """The URL the ``role`` (llm, stt, tts, realtime, embeddings) client of ``provider`` connects to for ``url``."""
    if provider == ServiceProviders.DEEPGRAM.value:
        return deepgram_client_url(url)
    if provider == ServiceProviders.ELEVENLABS.value:
        if role == "stt":
            return f"wss://{elevenlabs_realtime_stt_host(url)}"
        return elevenlabs_websocket_url(url)
    return url


def config_client_urls(sections: Mapping[str, Any]) -> list[str]:
    """The client URLs of the URLs the tenant set in these config sections, by role; a None section is skipped."""
    return [
        service_client_url(getattr(config, "provider", None), role, url)
        for role, config in sections.items()
        if config is not None
        for field in SERVICE_URL_FIELDS
        if isinstance(url := getattr(config, field, None), str) and url
    ]
