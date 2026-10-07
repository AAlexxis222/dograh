import importlib
from pathlib import Path

import pytest

from api import constants
from api.services import posthog_client

API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = API_DIR.parent


@pytest.fixture
def reload_telemetry(monkeypatch):
    """Reload constants + posthog_client under the test's env, then restore both."""
    created = []

    class FakePosthog:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

        def capture(self, **kwargs):
            pass

    def reload():
        importlib.reload(constants)
        importlib.reload(posthog_client)
        monkeypatch.setattr(posthog_client, "Posthog", FakePosthog)
        monkeypatch.setattr(posthog_client, "_posthog_client", None)
        return created

    yield reload

    monkeypatch.undo()
    importlib.reload(constants)
    importlib.reload(posthog_client)


def test_telemetry_defaults_to_false(monkeypatch, reload_telemetry):
    monkeypatch.delenv("ENABLE_TELEMETRY", raising=False)
    reload_telemetry()
    assert constants.ENABLE_TELEMETRY is False


def test_posthog_client_is_not_built_without_opt_in(monkeypatch, reload_telemetry):
    monkeypatch.delenv("ENABLE_TELEMETRY", raising=False)
    monkeypatch.setenv("POSTHOG_API_KEY", "phc_test")
    created = reload_telemetry()

    assert posthog_client.get_posthog() is None
    posthog_client.capture_event("user-1", "call_started", {"a": 1})
    assert created == []


def test_posthog_client_is_built_with_opt_in_and_key(monkeypatch, reload_telemetry):
    monkeypatch.setenv("ENABLE_TELEMETRY", "true")
    monkeypatch.setenv("POSTHOG_API_KEY", "phc_test")
    created = reload_telemetry()

    posthog_client.capture_event("user-1", "call_started", {"a": 1})
    assert len(created) == 1


def test_sentry_never_sends_default_pii():
    src = (API_DIR / "app.py").read_text(encoding="utf-8")
    assert "send_default_pii=True" not in src


def test_compose_embeds_no_posthog_key_and_defaults_telemetry_off():
    compose = (REPO_ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    assert "phc_" not in compose
    assert "${ENABLE_TELEMETRY:-true}" not in compose
