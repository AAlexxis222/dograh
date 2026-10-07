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
# Services that share the api cell env anchor; VOZ-N0-08 extends this set when it adds the call/arq/coordinator roles.
CELL_ROLES = {"api"}
AWS_CONFIG_TARGET = "/etc/xpand/aws/config"
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
    assert not env.get("AWS_ACCESS_KEY_ID") and not env.get("AWS_SECRET_ACCESS_KEY")  # host-vended short-lived credentials (credential_process), never IMDS or static keys
    assert not any(k.startswith("OTEL_EXPORTER") for k in env)  # spans off in cell
    # The env-level trace exporter is the Langfuse triple (api/constants.py); it must stay unset.
    assert not [k for k in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if env.get(k)]


def test_recordings_go_to_s3_not_minio(cfg):
    # The MinIO bucket carries an anonymous read/write policy (api/services/filesystem/minio.py).
    env = cfg["api"]["environment"]
    assert env["ENABLE_AWS_S3"] == "true"
    assert env["S3_BUCKET"] and env["S3_REGION"]


def test_api_reads_host_vended_credentials_never_imds(cfg):
    # The host vends short-lived credentials as a credential_process file (VOZ-AC-B5-55/-91); no IMDS, no static keys.
    env = cfg["api"]["environment"]
    assert env["AWS_CONFIG_FILE"] == AWS_CONFIG_TARGET
    assert env["AWS_SHARED_CREDENTIALS_FILE"] == "/dev/null"
    assert env["AWS_EC2_METADATA_DISABLED"] == "true"
    assert env["AWS_ACCESS_KEY_ID"] == "" and env["AWS_SECRET_ACCESS_KEY"] == ""
    assert env["AWS_SESSION_TOKEN"] == ""  # the base passes it through from .env; pipecat's AWS helpers read it


def _config_mounts(service):
    return [c for c in service.get("configs", []) if c["target"] == AWS_CONFIG_TARGET]


def test_api_ships_the_aws_config_from_the_overlay(cfg):
    # botocore reads AWS_CONFIG_FILE once at import; a host-written file could appear after api starts and stay empty.
    mounts = _config_mounts(cfg["api"])
    assert len(mounts) == 1, cfg["api"].get("configs")
    assert mounts[0]["source"] == "cell-aws-config"


def test_aws_config_content_is_the_credential_process(empty_env_file):
    _require_docker()
    out = _render(empty_env_file, {k: "x" for k in _required_names()}, "--format", "json")
    assert out.returncode == 0, out.stderr
    content = json.loads(out.stdout)["configs"]["cell-aws-config"]["content"]
    assert "credential_process = /bin/cat /run/aws/credentials.json" in content


@pytest.mark.parametrize("role", sorted(CELL_ROLES))
def test_cell_role_with_aws_env_has_both_mounts(role, cfg):
    # A role that takes the env anchor but forgets a mount would silently lose credentials.
    service = cfg[role]
    assert service["environment"].get("AWS_CONFIG_FILE") == AWS_CONFIG_TARGET
    assert len(_config_mounts(service)) == 1
    mounts = [v for v in service["volumes"] if v["target"] == "/run/aws"]
    assert len(mounts) == 1 and mounts[0].get("read_only") is True


def test_no_service_has_aws_env_outside_cell_roles(cfg):
    assert {n for n, s in cfg.items() if s.get("environment", {}).get("AWS_CONFIG_FILE")} == CELL_ROLES


def test_api_mounts_credentials_directory_read_only(cfg):
    mounts = [v for v in cfg["api"]["volumes"] if v["target"] == "/run/aws"]
    assert len(mounts) == 1, mounts
    mount = mounts[0]
    assert mount["type"] == "bind" and mount.get("read_only") is True
    # Docker on Windows may rewrite the host path (drive letter, backslashes); only that is tolerated.
    assert mount["source"].replace("\\", "/").endswith("/run/xpand/aws/recordings"), mount["source"]
    assert mount["bind"].get("create_host_path") is False


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
