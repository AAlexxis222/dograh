"""VOZ-AT-B5-08a: static hardening of the cell overlay, evaluated on `docker compose config`."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Every service the base defines under the profiles below, minus cloudflared (the cell never tunnels).
EXPECTED_SERVICES = {"postgres", "redis", "minio", "dograh-init", "nginx", "coturn", "api", "ui"}
LONG_LIVED = EXPECTED_SERVICES - {"dograh-init"}  # dograh-init is a one-shot container
INTERNAL_ONLY = {"postgres", "redis", "api", "ui", "minio"}
# Real profile names of the base file; the tunnel one is on to prove the overlay still drops cloudflared.
PROFILES = "remote,local-turn,tunnel"
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
def empty_env_file(tmp_path_factory):
    """An empty --env-file: a developer's repo-root .env must not leak into the rendering."""
    path = tmp_path_factory.mktemp("cell-env") / "empty.env"
    path.write_text("")
    return path


def _render(env_file, values, *extra):
    """Run `docker compose config` with only `values` (plus platform vars and profiles) in scope."""
    env = {**values, **_platform_env(), "COMPOSE_PROFILES": PROFILES}
    return subprocess.run(
        [*COMPOSE, "--env-file", str(env_file), "config", *extra], cwd=ROOT, env=env, capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def cfg(empty_env_file):
    _require_docker()
    out = _render(empty_env_file, {k: "x" for k in _required_names()}, "--format", "json")
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)["services"]


def test_no_internal_service_publishes_ports(cfg):
    assert {s for s in INTERNAL_ONLY if cfg[s].get("ports")} == set()


def test_expected_services_are_rendered_and_cloudflared_is_not(cfg):
    assert set(cfg) == EXPECTED_SERVICES  # also fails if the base grows a service the overlay does not harden


def test_every_service_rotates_logs(cfg):
    for name, service in cfg.items():
        assert service.get("logging", {}).get("options", {}).get("max-size"), name


def test_long_lived_services_restart_unless_stopped(cfg):
    for name in LONG_LIVED:
        assert cfg[name].get("restart") == "unless-stopped", name


def test_privacy_and_security_env(cfg):
    env = cfg["api"]["environment"]
    assert env["ENABLE_TELEMETRY"] == "false" and not env.get("POSTHOG_API_KEY")
    assert env["TELEPHONY_WS_TOKEN_ENFORCE"] == "true" and env["ENABLE_SIGNUP"] == "false"
    assert env["LOG_LEVEL"] == "INFO" and env["SERIALIZE_LOG_OUTPUT"] == "true" and "LOG_FILE_PATH" not in env
    assert env["URL_GUARD_ENFORCE"] == "true"
    assert not env.get("AWS_ACCESS_KEY_ID") and not env.get("AWS_SECRET_ACCESS_KEY")  # S3 by instance role
    assert not any(k.startswith("OTEL_EXPORTER") for k in env)  # spans off in cell
    # The env-level trace exporter is the Langfuse triple (api/constants.py); it must stay unset.
    assert not [k for k in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if env.get(k)]


def test_recordings_go_to_s3_not_minio(cfg):
    # The MinIO bucket carries an anonymous read/write policy (api/services/filesystem/minio.py).
    env = cfg["api"]["environment"]
    assert env["ENABLE_AWS_S3"] == "true"
    assert env["S3_BUCKET"] and env["S3_REGION"]


def test_ui_has_no_telemetry(cfg):
    env = cfg["ui"]["environment"]
    assert env["ENABLE_TELEMETRY"] == "false" and not env.get("POSTHOG_KEY")


def test_redis_has_aof(cfg):
    command = " ".join(cfg["redis"]["command"])
    assert "--appendonly yes" in command and "--appendfsync everysec" in command


@pytest.mark.parametrize("missing", _required_names())
def test_each_secret_is_required(missing, empty_env_file):
    _require_docker()
    others = {k: "x" for k in _required_names() if k != missing}
    out = _render(empty_env_file, others)
    assert out.returncode != 0 and missing in out.stderr, out.stderr
