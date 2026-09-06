"""Transport-level capture helpers for service tuning tests.

The factory tests in api/tests/test_*_service_factory.py assert on the
Settings object handed to a service (layer 2). Provider knobs are lost in a
third layer — the service drops them when it serialises the connection
(Flux language hints on a non-multi model, ElevenLabs language on a
non-multilingual model, Dograh TTS pitch/volume). These helpers capture what
actually leaves the process so tuning tests assert on the wire.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlsplit

from api.services.pipecat.audio_config import AudioConfig


def user_config_stt(
    provider: str,
    *,
    model: str,
    api_key: str = "test-key",
    language=None,
    base_url=None,
    **fields,
):
    return SimpleNamespace(
        stt=SimpleNamespace(
            provider=provider,
            model=model,
            api_key=api_key,
            language=language,
            base_url=base_url,
            **fields,
        )
    )


def user_config_tts(
    provider: str,
    *,
    model: str,
    voice: str = "Name - voice-1",
    api_key: str = "test-key",
    speed: float = 1.0,
    base_url=None,
    **fields,
):
    return SimpleNamespace(
        tts=SimpleNamespace(
            provider=provider,
            model=model,
            voice=voice,
            api_key=api_key,
            speed=speed,
            base_url=base_url,
            **fields,
        )
    )


def audio_config(rate: int = 16000) -> AudioConfig:
    return AudioConfig(transport_in_sample_rate=rate, transport_out_sample_rate=rate)


def query_params(url: str) -> dict[str, list[str]]:
    return parse_qs(urlsplit(url).query, keep_blank_values=True)


async def capture_ws_connect(monkeypatch, service) -> dict[str, Any]:
    """Drive ``service._connect()`` with the websockets client faked.

    Works for the Flux STT services (Deepgram, Dograh) exercised by the
    control tests. Both go through ``pipecat.services.websocket_service
    .websocket_connect`` (websocket_service.py:119), but two things the
    original design assumed don't hold for them and are worked around here:

    - Their ``_connect_websocket()`` asserts ``_websocket_url`` is already
      set (deepgram/flux/stt.py:315); that URL is only built inside the
      concrete ``_connect()`` override (deepgram/flux/stt.py:279-281,
      dograh/flux/stt.py analogous), so this drives ``_connect()`` rather
      than calling ``_connect_websocket()`` directly.
    - After connecting, they await ``_connection_established_event``, set
      only once the receive task observes a "Connected" message from the
      real server (deepgram/flux/base.py:204,617). No real server exists
      here, so ``create_task`` is stubbed to drop the receive/watchdog
      coroutines (closing them to avoid a "never awaited" RuntimeWarning,
      mirroring TaskManager.create_task's own cleanup at
      task_manager.py:196-207) and the fake connect sets the event itself,
      once it stands in for the real connection, so the wait resolves
      immediately instead of hanging. (The event is cleared right before the
      connect call inside ``_connect_websocket()``, so arming it any earlier
      would just be wiped.)
    """
    captured: dict[str, Any] = {}

    async def fake_connect(url, *args, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("additional_headers", {})
        ws = MagicMock()
        ws.response = SimpleNamespace(headers={})
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        # `_connect_websocket()` clears `_connection_established_event` right
        # before opening the connection (deepgram/flux/stt.py:310-311) and
        # only awaits it afterwards, so pre-arming before `_connect()` is
        # called would just get wiped. Set it here instead, once this fake
        # stands in for the real connection.
        connected_event = getattr(service, "_connection_established_event", None)
        if connected_event is not None:
            connected_event.set()
        return ws

    def fake_create_task(coro, *args, **kwargs):
        coro.close()
        return None

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_connect
    )
    monkeypatch.setattr(service, "create_task", fake_create_task)

    await service._connect()
    return captured


async def capture_nova_connect(service) -> dict[str, Any]:
    """Return the kwargs DeepgramSTTService passes to ``listen.v1.connect``.

    ``_connection_handler`` (deepgram/stt.py:643-659) builds them from
    ``_build_connect_kwargs``; we call the builder directly because the
    handler is a long-lived task.
    """
    return service._build_connect_kwargs()


class FakeWebSocket:
    def __init__(self):
        from websockets.protocol import State

        self.state = State.OPEN
        self.sent: list[dict] = []

    async def send(self, data: str):
        self.sent.append(json.loads(data))


async def capture_elevenlabs_context_init(service) -> dict[str, Any]:
    """Drive ``run_tts`` once with a fake socket and return the context-init
    message (elevenlabs/tts.py:1050-1062), which carries ``voice_settings``.

    ``run_tts(text, context_id)`` requires an explicit ``context_id``
    (elevenlabs/tts.py:1022), and yields ``TTSStartedFrame`` *before* it
    sends the context-init message (elevenlabs/tts.py:1045-1062) — the
    context-init send only happens once the generator is resumed past that
    first yield. So the generator is drained fully rather than advanced a
    single step.
    """
    ws = FakeWebSocket()
    service._websocket = ws
    agen = service.run_tts("hello", context_id="ctx-1")
    try:
        async for _ in agen:
            pass
    finally:
        await agen.aclose()
    return ws.sent[0]


def chat_payload(service) -> dict[str, Any]:
    return service.build_chat_completion_params(
        {"messages": [{"role": "user", "content": "hi"}]}
    )


def responses_payload(service) -> dict[str, Any]:
    return service._build_response_params(
        {"input": [{"role": "user", "content": "hi"}]}
    )


async def capture_openai_tts_request(service, text: str = "hi") -> dict[str, Any]:
    """Drive ``run_tts`` once with a faked streaming response and return the
    kwargs passed to ``audio.speech.with_streaming_response.create``.

    ``run_tts(text, context_id)`` requires an explicit ``context_id``
    (openai/tts.py:244) and checks ``r.status_code`` before reading the
    stream (openai/tts.py:283), so the fake response needs both.
    """
    create = MagicMock()

    async def _empty(*args, **kwargs):
        return
        yield b""  # pragma: no cover - makes this an async generator

    @asynccontextmanager
    async def _streaming(**kwargs):
        create(**kwargs)
        yield SimpleNamespace(status_code=200, iter_bytes=_empty)

    service._client.audio.speech.with_streaming_response.create = _streaming
    agen = service.run_tts(text, context_id="ctx-1")
    try:
        async for _ in agen:
            pass
    finally:
        await agen.aclose()
    return create.call_args.kwargs
