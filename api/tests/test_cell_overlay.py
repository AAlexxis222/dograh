"""VOZ-AT-B5-08a: static hardening of the cell overlay, evaluated on `docker compose config`."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Every service the base defines under the profiles below, minus cloudflared (the cell never tunnels).
EXPECTED_SERVICES = {"postgres", "redis", "minio", "dograh-init", "nginx", "coturn", "api", "call", "arq", "coordinators", "ui"}
LONG_LIVED = EXPECTED_SERVICES - {"dograh-init"}  # dograh-init is a one-shot container
INTERNAL_ONLY = {"postgres", "redis", "api", "call", "arq", "coordinators", "ui", "minio"}
# Real profile names of the base file; the tunnel one is on to prove the overlay still drops cloudflared.
PROFILES = "remote,local-turn,tunnel"
# Services that share the api cell env anchor: the roles run from one image (VOZ-N0-08).
CELL_ROLES = {"api", "call", "arq", "coordinators"}
# Not a secret: any memory size the cell host sets; the placeholder "x" used for the secrets is not a valid size.
CELL_API_IMAGE = "registry.example/xpand-api@sha256:" + "0" * 64  # the fork image, pinned by digest
VALID_VALUES = {"CALL_MEM_LIMIT": "1g", "CELL_API_IMAGE": CELL_API_IMAGE}
AWS_CONFIG_TARGET = "/etc/xpand/aws/config"
COMPOSE = ["docker", "compose", "-f", "docker-compose.yaml", "-f", "docker-compose.cell.yaml"]
# Docker on Windows needs these to find its compose plugin (ProgramFiles) and config/context; none is a secret.
PLATFORM_VARS = ("PATH", "SystemRoot", "ProgramFiles", "ProgramData", "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA")


def _required_names():
    lines = (ROOT / "deploy/cell/.env.reference").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _values(skip=None):
    """A value for every required variable except `skip`: "x" for secrets, a real size where one is parsed."""
    return {k: VALID_VALUES.get(k, "x") for k in _required_names() if k != skip}


def _platform_env():
    return {k: os.environ[k] for k in PLATFORM_VARS if k in os.environ}


def _require_docker():
    if not shutil.which("docker"):
        pytest.skip("GATED docker_missing: install Docker Compose >= 2.26.1 (2.24.6 to 2.25.0 reject this overlay)")


def _compose_major():
    out = subprocess.run([*COMPOSE[:2], "version", "--short"], env=_platform_env(), capture_output=True, text=True)
    return int(out.stdout.strip().lstrip("v").split(".")[0])


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
    out = _render(empty_env_file, _values(), "--format", "json")
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
    out = _render(empty_env_file, _values(), "--format", "json")
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
    # A missing credentials directory must fail the start, never be created empty. Compose 2.x renders create_host_path: false as `bind: {}` (true as an explicit true); 5.x the reverse.
    if _compose_major() >= 5:
        assert mount["bind"].get("create_host_path") is False
    else:
        assert mount.get("bind") is not None and "create_host_path" not in mount["bind"]


def test_coturn_is_on_its_own_network(cfg):
    # VOZ-AC-B4-05b: a relay in app-network could reach postgres, redis and api. The api mints TURN credentials locally.
    assert set(cfg["coturn"]["networks"]) == {"turn-net"}
    assert not [n for n, s in cfg.items() if n != "coturn" and "turn-net" in s.get("networks", {})]


def test_turn_init_denies_internal_peers_even_on_a_private_address(cfg):
    # Fail closed in a cell (VOZ-AC-B0-30): the render would otherwise skip the deny block for a private SERVER_IP.
    assert cfg["dograh-init"]["environment"]["TURN_DENY_INTERNAL_PEERS"] == "true"


def test_turn_init_always_takes_the_render_branch(cfg):
    # Outside production dograh-init no-ops without TURN_HOST, coturn then starts on its defaults (an open relay).
    # The remote branch renders the config and fails closed on a missing SERVER_IP or certs.
    assert cfg["dograh-init"]["environment"]["ENVIRONMENT"] == "production"


def test_ui_has_no_telemetry(cfg):
    env = cfg["ui"]["environment"]
    assert env["ENABLE_TELEMETRY"] == "false" and not env.get("POSTHOG_KEY")


def test_redis_has_aof(cfg):
    command = " ".join(cfg["redis"]["command"])
    assert "--appendonly yes" in command and "--appendfsync everysec" in command


@pytest.mark.parametrize("missing", _required_names())
def test_each_secret_is_required(missing, empty_env_file):
    _require_docker()
    others = _values(skip=missing)
    out = _render(empty_env_file, others)
    assert out.returncode != 0 and missing in out.stderr, out.stderr


def test_every_role_runs_the_fork_image_from_one_variable(cfg):
    # The call entrypoint and the migration gates exist only in the fork image: upstream's default image would crash-loop.
    assert {r: cfg[r]["image"] for r in CELL_ROLES} == {r: CELL_API_IMAGE for r in CELL_ROLES}


@pytest.mark.parametrize("role", sorted(CELL_ROLES))
def test_migrations_are_off_the_startup_path_in_every_role(role, cfg):
    assert cfg[role]["environment"]["RUN_MIGRATIONS_ON_START"] == "false"


def test_call_role_drains_and_is_the_oom_victim(cfg):
    call = cfg["call"]
    assert "call_entrypoint.sh" in " ".join(call["entrypoint"])
    assert call["mem_limit"] == str(1024**3)  # CALL_MEM_LIMIT=1g
    assert call["oom_score_adj"] == 500
    assert "stop_grace_period" not in call  # VOZ-N0-20 renders it from durations.py


def test_each_duty_runs_in_exactly_one_role(cfg):
    def env(role):
        return cfg[role]["environment"]

    # Every other role runs no coordinators and no arq worker inside its start script.
    for role in CELL_ROLES - {"coordinators"}:
        assert env(role)["ENABLE_ARI_MANAGER"] == "false" and env(role)["ENABLE_CAMPAIGN_ORCHESTRATOR"] == "false", role
        assert env(role)["ARQ_WORKERS"] == "0", role
    # coordinators runs the two singletons and no uvicorn / arq.
    assert env("coordinators")["ENABLE_ARI_MANAGER"] == "true" and env("coordinators")["ENABLE_CAMPAIGN_ORCHESTRATOR"] == "true"
    assert env("coordinators")["FASTAPI_WORKERS"] == "0" and env("coordinators")["ARQ_WORKERS"] == "0"
    assert "run_arq_worker.sh" in " ".join(cfg["arq"]["command"])


@pytest.mark.parametrize("role", sorted(CELL_ROLES))
def test_each_role_names_itself_for_the_schema_gate(role, cfg):
    # The gate's failure line reports where=$CELL_ROLE, so operators can tell which role refused to start.
    assert cfg[role]["environment"]["CELL_ROLE"] == role


def test_arq_refuses_a_stale_schema_before_it_starts(cfg):
    # VOZ-AC-B5-44: the worker is not launched through the base start script, so its command carries the gate.
    command = " ".join(cfg["arq"]["command"])
    # `&&` chains the two: with `;` the worker would start after a refused gate.
    assert re.search(r"require_db_head\.sh\s+arq\s*&&\s*(exec\s+)?\S*run_arq_worker\.sh", command), command


def test_image_ships_the_call_entrypoint():
    # The Dockerfile copies scripts by allowlist; a missing line leaves the call role without its entrypoint.
    dockerfile = (ROOT / "api/Dockerfile").read_text()
    assert "scripts/xpand/call_entrypoint.sh" in dockerfile
    assert "scripts/xpand/require_db_head.sh" in dockerfile
