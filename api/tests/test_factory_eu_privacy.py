"""VOZ-N0-26: the platform's defaults keep call audio in the EU and out of training.

Caller audio can carry health data, so what a call sends and where it goes is
decided at the source of truth: the registry fills the endpoint before the
factory ever sees the configuration (a fallback in the factory would pass its
own tests and change no real call). Each test names the assertion that goes
red when the change is removed; the ones that cannot (pipecat already does it)
are labelled "regression".

``mip_opt_out`` is a policy, not a choosable default, so it is fixed in the
factory and the tuning allow-list refuses it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.configuration.registry import (
    AWS_NOVA_SONIC_REGIONS,
    AWSNovaSonicRealtimeLLMConfiguration,
    DeepgramSTTConfiguration,
    DeepgramTTSConfiguration,
)
from api.services.pipecat.service_factory import (
    create_llm_service_from_provider,
    create_realtime_llm_service,
    create_stt_service,
    create_tts_service,
)
from api.tests.service_tuning._transport import (
    audio_config,
    capture_nova_connect,
    capture_ws_connect,
    query_params,
    user_config_stt,
)
from api.tests.test_openai_reasoning_declared import _one_turn, _responses_tuning

EU = "https://api.eu.deepgram.com"
EU_WS = "wss://api.eu.deepgram.com"


def _stt_user_config(**fields):
    return SimpleNamespace(stt=DeepgramSTTConfiguration(api_key="test-key", **fields))


def _tts_user_config():
    return SimpleNamespace(tts=DeepgramTTSConfiguration(api_key="test-key"))


def _ctor_kwargs(service_name: str, build):
    """The kwargs the factory hands to one pipecat service class."""
    with patch(f"api.services.pipecat.service_factory.{service_name}") as mock:
        build()
    mock.assert_called_once()
    return mock.call_args.kwargs


# --- registry defaults: the value the factory actually receives --------------


@pytest.mark.parametrize(
    "config_cls", [DeepgramSTTConfiguration, DeepgramTTSConfiguration]
)
def test_deepgram_registry_default_is_the_eu_host(config_cls):
    # Red at step 2: the default is https://api.deepgram.com.
    assert config_cls(api_key="k").base_url == EU


def test_nova_sonic_registry_default_region_is_eu_north_1():
    # Red at step 2: the default region is us-east-1.
    config = AWSNovaSonicRealtimeLLMConfiguration(
        aws_access_key="a", aws_secret_key="s"
    )
    assert config.aws_region == "eu-north-1"
    assert config.aws_region in AWS_NOVA_SONIC_REGIONS


def test_nova_sonic_default_region_reaches_the_service():
    # Red at step 2: the service is built with us-east-1.
    user_config = SimpleNamespace(
        realtime=AWSNovaSonicRealtimeLLMConfiguration(
            aws_access_key="a", aws_secret_key="s"
        )
    )
    with patch(
        "api.services.pipecat.realtime.aws_nova_sonic.DograhAWSNovaSonicLLMService"
    ) as mock:
        create_realtime_llm_service(user_config, audio_config())
    assert mock.call_args.kwargs["region"] == "eu-north-1"


# --- the configuration the registry fills reaches the constructors -----------


def test_nova_built_from_registry_defaults_uses_eu_host_and_opts_out():
    # Red at step 2: base_url is the US host and mip_opt_out is not passed.
    kwargs = _ctor_kwargs(
        "DeepgramSTTService",
        lambda: create_stt_service(_stt_user_config(), audio_config()),
    )
    assert kwargs["base_url"] == EU
    assert kwargs["mip_opt_out"] is True


def test_flux_built_from_registry_defaults_uses_eu_socket_and_opts_out():
    # Red at step 2: url is wss://api.deepgram.com/v2/listen, no mip_opt_out.
    kwargs = _ctor_kwargs(
        "DeepgramFluxSTTService",
        lambda: create_stt_service(
            _stt_user_config(model="flux-general-multi"), audio_config()
        ),
    )
    assert kwargs["url"] == f"{EU_WS}/v2/listen"
    assert kwargs["mip_opt_out"] is True


def test_deepgram_tts_built_from_registry_defaults_uses_eu_socket_and_opts_out():
    # Red at step 2: base_url is wss://api.deepgram.com, no mip_opt_out.
    kwargs = _ctor_kwargs(
        "DeepgramTTSService",
        lambda: create_tts_service(_tts_user_config(), audio_config()),
    )
    assert kwargs["base_url"] == EU_WS
    assert kwargs["mip_opt_out"] is True


def test_a_stored_us_host_is_still_honoured():
    # Regression: an operator who chose the US host explicitly keeps it; the
    # change moves the default, it does not override a choice.
    kwargs = _ctor_kwargs(
        "DeepgramSTTService",
        lambda: create_stt_service(
            _stt_user_config(base_url="https://api.deepgram.com"), audio_config()
        ),
    )
    assert kwargs["base_url"] == "https://api.deepgram.com"


# --- on the wire --------------------------------------------------------------


async def test_nova_connect_sends_mip_opt_out_true():
    # Red at step 2: the connect kwargs carry no mip_opt_out.
    service = create_stt_service(_stt_user_config(), audio_config())
    assert (await capture_nova_connect(service))["mip_opt_out"] == "true"


async def test_flux_socket_is_eu_and_sends_mip_opt_out_true(monkeypatch):
    # Red at step 2: the socket is api.deepgram.com and the query has no
    # mip_opt_out.
    service = create_stt_service(
        _stt_user_config(model="flux-general-multi"), audio_config()
    )
    captured = await capture_ws_connect(monkeypatch, service)
    assert captured["url"].startswith(f"{EU_WS}/v2/listen?")
    assert query_params(captured["url"])["mip_opt_out"] == ["true"]


async def test_deepgram_tts_socket_is_eu_and_sends_mip_opt_out_true(monkeypatch):
    # Red at step 2: the socket is api.deepgram.com and the query has no
    # mip_opt_out.
    service = create_tts_service(_tts_user_config(), audio_config())
    seen = {}

    async def fake_connect(url, **kwargs):
        seen["url"] = url
        ws = MagicMock()
        ws.response = SimpleNamespace(headers={})
        ws.close = AsyncMock()
        return ws

    monkeypatch.setattr(service, "_websocket_connect", fake_connect)
    service._call_event_handler = AsyncMock()
    await service._connect_websocket()
    assert seen["url"].startswith(f"{EU_WS}/v1/speak?")
    assert query_params(seen["url"])["mip_opt_out"] == ["true"]


async def test_dograh_flux_sends_mip_opt_out_true(monkeypatch):
    # Red at step 2: the query has no mip_opt_out.
    service = create_stt_service(
        user_config_stt("dograh", model="ignored", language="en"),
        audio_config(),
        correlation_id="corr-1",
    )
    captured = await capture_ws_connect(monkeypatch, service)
    assert query_params(captured["url"])["mip_opt_out"] == ["true"]


# --- tuning cannot turn it back off -------------------------------------------


@pytest.mark.parametrize("provider", ["deepgram", "dograh"])
def test_tuning_refuses_mip_opt_out_in_ctor(provider):
    # Red at step 2: the allow-list accepts ctor.mip_opt_out for both.
    with pytest.raises(
        ValidationError, match=f"stt.{provider}.ctor.mip_opt_out: not allowed"
    ):
        WorkflowConfigurationDefaults.model_validate(
            {"service_tuning": {"stt": {provider: {"ctor": {"mip_opt_out": False}}}}}
        )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda tuning: create_stt_service(
                _stt_user_config(model="flux-general-multi"),
                audio_config(),
                tuning=tuning,
            ),
            id="flux",
        ),
        pytest.param(
            lambda tuning: create_stt_service(
                user_config_stt("dograh", model="ignored", language="en"),
                audio_config(),
                tuning=tuning,
            ),
            id="dograh-flux",
        ),
    ],
)
async def test_a_stale_stored_false_cannot_reach_the_wire(monkeypatch, build):
    # Red at step 2: the stored false is splatted into the constructor, so the
    # query says mip_opt_out=false. A tuning saved before the allow-list
    # tightened skips validation, so the factory has to win on its own.
    stale = {
        "stt": {
            "deepgram": {"ctor": {"mip_opt_out": False}},
            "dograh": {"ctor": {"mip_opt_out": False}},
        }
    }
    service = build(stale)
    captured = await capture_ws_connect(monkeypatch, service)
    assert query_params(captured["url"])["mip_opt_out"] == ["true"]


@pytest.mark.parametrize("provider", ["deepgram", "dograh"])
def test_tuning_refuses_mip_opt_out_in_settings(provider):
    # Regression: neither Settings class declares it, so the second door is
    # shut today. Pinned because Nova folds unknown settings into ``extra``,
    # which pipecat sends after the constructor value and would win.
    with pytest.raises(
        ValidationError, match=f"stt.{provider}.settings.mip_opt_out: unknown"
    ):
        WorkflowConfigurationDefaults.model_validate(
            {
                "service_tuning": {
                    "stt": {provider: {"settings": {"mip_opt_out": False}}}
                }
            }
        )


# --- regressions (pipecat already does this; the test pins it) ----------------


def test_speechmatics_keeps_the_eu_agent_host(monkeypatch):
    # Regression: pipecat defaults to wss://eu2.rt.speechmatics.com/v2/agent
    # (speechmatics/stt.py:437); a SPEECHMATICS_RT_URL in the environment
    # would override it, so it is cleared here.
    monkeypatch.delenv("SPEECHMATICS_RT_URL", raising=False)
    service = create_stt_service(
        user_config_stt("speechmatics", model="enhanced", language="es"),
        audio_config(),
    )
    assert service._base_url.startswith("wss://eu")
    assert service._base_url.endswith("/v2/agent")


async def test_openai_responses_wire_sends_store_false():
    # Regression: pipecat sends store=False (openai/responses/llm.py:373) and
    # the tuning path does not overwrite it.
    llm = create_llm_service_from_provider(
        provider="openai",
        model="gpt-6-luna",
        api_key="k",
        tuning=_responses_tuning(),
    )
    assert (await _one_turn(llm))["store"] is False
