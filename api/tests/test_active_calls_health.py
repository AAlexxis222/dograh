"""VOZ-N0-22: a cell refuses to start without the devops secret, and active-calls reports the drain fields."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app import assert_cell_startup_config
from api.routes import main as main_routes

SECRET = "test-dograh-devops-secret"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("api.constants.DOGRAH_DEVOPS_SECRET", SECRET)
    app = FastAPI()
    app.add_api_route(
        "/api/v1/health/active-calls",
        main_routes.active_calls,
        methods=["GET"],
        response_model=main_routes.ActiveCallsResponse,
    )
    return TestClient(app)


def _get(client):
    return client.get(
        "/api/v1/health/active-calls",
        headers={main_routes.DOGRAH_DEVOPS_SECRET_HEADER: SECRET},
    )


def test_active_calls_reports_drain_fields(client):
    body = _get(client).json()
    assert {"active_calls", "lag_p99_ms", "draining", "k_p"} <= body.keys()
    assert body["draining"] is False
    assert body["k_p"] == 4


def test_draining_follows_the_flag_file(client, monkeypatch, tmp_path):
    flag = tmp_path / "draining"
    monkeypatch.setenv("DRAIN_FLAG_FILE", str(flag))
    assert _get(client).json()["draining"] is False
    flag.touch()
    assert _get(client).json()["draining"] is True


def test_k_p_comes_from_env_and_falls_back_outside_a_cell(client, monkeypatch):
    monkeypatch.setenv("CALL_K_P", "7")
    assert _get(client).json()["k_p"] == 7
    monkeypatch.setenv("CALL_K_P", "zero")
    assert _get(client).json()["k_p"] == 4


@pytest.fixture
def cell_env(monkeypatch):
    monkeypatch.setenv("CELL_ROLE", "call")
    monkeypatch.setenv("DOGRAH_DEVOPS_SECRET", SECRET)
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("CALL_K_P", raising=False)


def test_startup_passes_with_a_valid_cell_config(cell_env):
    assert_cell_startup_config()


def test_startup_fails_named_without_devops_secret_in_cell(cell_env, monkeypatch):
    monkeypatch.delenv("DOGRAH_DEVOPS_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="code=devops_secret_missing") as e:
        assert_cell_startup_config()
    for key in ("where=", "reason=", "hint="):
        assert key in str(e.value)


def test_startup_fails_closed_on_empty_devops_secret_in_cell(cell_env, monkeypatch):
    monkeypatch.setenv("DOGRAH_DEVOPS_SECRET", "")
    with pytest.raises(RuntimeError, match="code=devops_secret_missing"):
        assert_cell_startup_config()


@pytest.mark.parametrize("level", ["DEBUG", "debug"])
def test_startup_refuses_debug_logs_in_cell(cell_env, monkeypatch, level):
    monkeypatch.setenv("LOG_LEVEL", level)
    with pytest.raises(RuntimeError, match="code=log_level_debug_in_cell"):
        assert_cell_startup_config()


def test_startup_refuses_unset_log_level_in_cell(cell_env, monkeypatch):
    # api/constants.py defaults an unset LOG_LEVEL to DEBUG.
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    with pytest.raises(RuntimeError, match="code=log_level_debug_in_cell"):
        assert_cell_startup_config()


@pytest.mark.parametrize("value", ["abc", "0", "-3"])
def test_startup_refuses_bad_k_p_in_cell(cell_env, monkeypatch, value):
    monkeypatch.setenv("CALL_K_P", value)
    with pytest.raises(RuntimeError, match="code=call_k_p_invalid"):
        assert_cell_startup_config()


def test_startup_reports_incoherent_durations_in_cell(cell_env, monkeypatch):
    monkeypatch.setenv("CELL_CALL_DURATION_CEILING_S", "1")
    with pytest.raises(RuntimeError, match="code=knob_out_of_range"):
        assert_cell_startup_config()


def test_startup_is_a_noop_outside_a_cell(monkeypatch):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    monkeypatch.delenv("DOGRAH_DEVOPS_SECRET", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert_cell_startup_config()
