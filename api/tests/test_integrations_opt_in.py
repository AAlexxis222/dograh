"""Observability integrations stay off unless the org itself opted in.

A deployment-wide environment variable must never be what sends a client's
calls to a third party: with several clients in one deployment, that would
export the transcripts of every client that never agreed to it.
"""

from types import SimpleNamespace
from typing import ClassVar

import pytest

from api.services.integrations import registry as integrations
from api.services.integrations.noveum.node import NoveumNodeData
from api.services.integrations.paygent.node import PaygentNodeData
from api.services.integrations.tuner.node import TunerNodeData
from api.services.pipecat import tracing_config

ENV_HOST = "https://env-langfuse.example.com"
ORG_HOST = "https://org-langfuse.example.com"

# Every variable an operator could set for these integrations.
DEPLOYMENT_ENV = {
    "LANGFUSE_HOST": ENV_HOST,
    "LANGFUSE_PUBLIC_KEY": "pk-env",
    "LANGFUSE_SECRET_KEY": "sk-env",
    "LANGFUSE_PROJECT_ID": "env-project",
    "NOVEUM_API_KEY": "noveum-env",
    "NOVEUM_PROJECT": "env-project",
}

EXPLICIT_ORG_ENTRIES = {
    "langfuse": {
        "host": ORG_HOST,
        "public_key": "pk-org",
        "secret_key": "sk-org",
        "project_id": "org-project",
    },
    "noveum": NoveumNodeData.model_validate(
        {"name": "Noveum", "noveum_api_key": "k", "noveum_project": "p"}
    ),
    "paygent": PaygentNodeData.model_validate(
        {
            "name": "Paygent",
            "paygent_api_key": "k",
            "paygent_agent_id": "a",
            "paygent_customer_id": "c",
            "paygent_indicator": "i",
        }
    ),
    "tuner": TunerNodeData.model_validate(
        {
            "name": "Tuner",
            "tuner_api_key": "k",
            "tuner_agent_id": "a",
            "tuner_workspace_id": 1,
        }
    ),
}


@pytest.fixture
def deployment_env(monkeypatch):
    for var, value in DEPLOYMENT_ENV.items():
        monkeypatch.setenv(var, value)


@pytest.mark.parametrize("name", sorted(EXPLICIT_ORG_ENTRIES))
def test_integration_inactive_without_explicit_org_config(name, deployment_env):
    assert integrations.is_active(name, None) is False


@pytest.mark.parametrize("name", sorted(EXPLICIT_ORG_ENTRIES))
def test_integration_active_with_explicit_org_config(name):
    assert integrations.is_active(name, EXPLICIT_ORG_ENTRIES[name]) is True


class _RecordingExporter:
    instances: ClassVar[list["_RecordingExporter"]] = []

    def __init__(self, endpoint, headers):
        self._endpoint = endpoint
        self._headers = headers
        self.exported = []
        _RecordingExporter.instances.append(self)

    def export(self, spans):
        self.exported.extend(spans)
        return tracing_config.SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def _span(org_id):
    attributes = {"dograh.org_id": org_id} if org_id else {}
    return SimpleNamespace(attributes=attributes, instrumentation_scope=None)


@pytest.fixture
def env_configured_tracing(monkeypatch):
    """Initialise tracing exactly as a deployment with Langfuse env vars would."""
    monkeypatch.setattr(_RecordingExporter, "instances", [])
    monkeypatch.setattr(tracing_config, "OTLPSpanExporter", _RecordingExporter)
    monkeypatch.setattr(tracing_config, "LANGFUSE_HOST", ENV_HOST)
    monkeypatch.setattr(tracing_config, "LANGFUSE_PROJECT_ID", "env-project")
    monkeypatch.setattr(tracing_config, "LANGFUSE_PUBLIC_KEY", "pk-env", raising=False)
    monkeypatch.setattr(tracing_config, "LANGFUSE_SECRET_KEY", "sk-env", raising=False)
    monkeypatch.setattr(tracing_config, "_tracing_initialized", False)
    monkeypatch.setattr(tracing_config, "_org_routing_exporter", None)
    monkeypatch.setattr(tracing_config, "setup_tracing", lambda **_: None)
    monkeypatch.setattr(tracing_config, "set_trace_public_resolver", lambda _: None)
    from opentelemetry import trace as otel_trace

    # Keep the global tracer provider untouched by this test.
    monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: object())

    assert tracing_config.ensure_tracing() is True
    return tracing_config._org_routing_exporter


def test_spans_of_a_non_opted_org_never_reach_the_env_langfuse(
    env_configured_tracing,
):
    routing = env_configured_tracing
    routing.register_org("opted", ORG_HOST, "pk-org", "sk-org", "org-project")
    opted, not_opted, unattributed = _span("opted"), _span("not-opted"), _span(None)

    routing.export([opted, not_opted, unattributed])

    env_exporters = [
        e for e in _RecordingExporter.instances if e._endpoint.startswith(ENV_HOST)
    ]
    assert all(not e.exported for e in env_exporters)
    # Positive control: the org that opted in still gets its own spans.
    org_exporters = [
        e for e in _RecordingExporter.instances if e._endpoint.startswith(ORG_HOST)
    ]
    assert [e.exported for e in org_exporters] == [[opted]]
