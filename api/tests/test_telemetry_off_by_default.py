import ast
import importlib
from pathlib import Path

import pytest
import yaml

from api import constants
from api.services import posthog_client

API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = API_DIR.parent


@pytest.fixture
def fake_posthog(monkeypatch):
    """Count PostHog client constructions instead of building a real one."""
    created = []

    class FakePosthog:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

        def capture(self, **kwargs):
            pass

    monkeypatch.setattr(posthog_client, "Posthog", FakePosthog)
    monkeypatch.setattr(posthog_client, "_posthog_client", None)
    monkeypatch.setattr(constants, "POSTHOG_API_KEY", "phc_test")
    return created


def test_telemetry_defaults_to_false(monkeypatch):
    monkeypatch.delenv("ENABLE_TELEMETRY", raising=False)
    try:
        assert importlib.reload(constants).ENABLE_TELEMETRY is False
    finally:
        monkeypatch.undo()
        importlib.reload(constants)


def test_posthog_client_is_not_built_without_opt_in(monkeypatch, fake_posthog):
    monkeypatch.setattr(constants, "ENABLE_TELEMETRY", False)

    assert posthog_client.get_posthog() is None
    posthog_client.capture_event("user-1", "call_started", {"a": 1})
    assert fake_posthog == []


def test_posthog_client_is_built_with_opt_in_and_key(monkeypatch, fake_posthog):
    monkeypatch.setattr(constants, "ENABLE_TELEMETRY", True)

    posthog_client.capture_event("user-1", "call_started", {"a": 1})
    assert len(fake_posthog) == 1


def test_sentry_never_sends_default_pii():
    tree = ast.parse((API_DIR / "app.py").read_text(encoding="utf-8"))
    inits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "init"
        and getattr(node.func.value, "id", None) == "sentry_sdk"
    ]
    assert inits, "sentry_sdk.init call not found in app.py"
    for call in inits:
        assert not any(kw.arg is None for kw in call.keywords), "no **kwargs"
        pii = [kw.value for kw in call.keywords if kw.arg == "send_default_pii"]
        assert len(pii) == 1
        assert isinstance(pii[0], ast.Constant) and pii[0].value is False


@pytest.mark.parametrize(
    "service,key_var", [("api", "POSTHOG_API_KEY"), ("ui", "POSTHOG_KEY")]
)
def test_compose_defaults_telemetry_off_without_embedded_key(service, key_var):
    text = (REPO_ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    assert "phc_" not in text
    env = yaml.safe_load(text)["services"][service]["environment"]
    assert env["ENABLE_TELEMETRY"] == "${ENABLE_TELEMETRY:-false}"
    assert env[key_var] == f"${{{key_var}:-}}"
    assert env["POSTHOG_HOST"].startswith("${POSTHOG_HOST:-")
