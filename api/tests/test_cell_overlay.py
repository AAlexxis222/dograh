"""VOZ-AT-B5-08a: static hardening of the cell overlay, evaluated on `docker compose config`."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LONG_LIVED = {"postgres", "redis", "minio", "nginx", "coturn", "api", "call", "arq", "coordinators", "ui"}
INTERNAL_ONLY = {"postgres", "redis", "api", "ui", "minio", "call"}
COMPOSE = ["docker", "compose", "-f", "docker-compose.yaml", "-f", "docker-compose.cell.yaml"]
# Docker on Windows needs these to find its compose plugin (ProgramFiles) and config/context; none is a secret.
PLATFORM_VARS = ("PATH", "SystemRoot", "ProgramFiles", "ProgramData", "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA")


def _required_names():
    lines = (ROOT / "deploy/cell/.env.reference").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _platform_env():
    return {k: os.environ[k] for k in PLATFORM_VARS if k in os.environ}


def _require_docker():
    if not shutil.which("docker"):
        pytest.skip("GATED docker_missing: install Docker Compose >= 2.24.4")


@pytest.fixture(scope="module")
def cfg():
    _require_docker()
    env = {**{k: "x" for k in _required_names()}, **_platform_env()}
    out = subprocess.run(
        [*COMPOSE, "config", "--format", "json"], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)["services"]


def test_no_internal_service_publishes_ports(cfg):
    assert {s for s in INTERNAL_ONLY & cfg.keys() if cfg[s].get("ports")} == set()


def test_long_lived_services_restart_and_rotate_logs(cfg):
    for name in LONG_LIVED & cfg.keys():
        assert cfg[name].get("restart") == "unless-stopped", name
        assert cfg[name].get("logging", {}).get("options", {}).get("max-size"), name


def test_privacy_and_security_env(cfg):
    env = cfg["api"]["environment"]
    assert env["ENABLE_TELEMETRY"] == "false" and not env.get("POSTHOG_API_KEY")
    assert env["TELEPHONY_WS_TOKEN_ENFORCE"] == "true" and env["ENABLE_SIGNUP"] == "false"
    assert env["LOG_LEVEL"] == "INFO" and env["SERIALIZE_LOG_OUTPUT"] == "true" and "LOG_FILE_PATH" not in env
    assert env["URL_GUARD_ENFORCE"] == "true"
    assert not env.get("AWS_ACCESS_KEY_ID") and not env.get("AWS_SECRET_ACCESS_KEY")  # S3 by instance role
    assert not any(k.startswith("OTEL_EXPORTER") for k in env)  # spans off in cell


def test_redis_has_aof(cfg):
    command = " ".join(cfg["redis"]["command"])
    assert "--appendonly yes" in command and "--appendfsync everysec" in command


def test_no_tunnel_in_cell(cfg):
    assert "cloudflared" not in cfg and "tunnel" not in cfg


def test_secrets_are_required():
    _require_docker()
    out = subprocess.run(
        [*COMPOSE, "config"], cwd=ROOT, env=_platform_env(), capture_output=True, text=True
    )
    assert out.returncode != 0 and "must be set" in out.stderr
