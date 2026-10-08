"""VOZ-AC-B4-05 / VOZ-AC-B4-05b: coturn is not a relay into the cell, and TURN credentials are short-lived."""

import os
import secrets
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "deploy/templates/turnserver.remote.conf.template"
SETUP_COMMON = ROOT / "scripts/lib/setup_common.sh"
RELAY_PORTS = range(49152, 49201)  # min-port / max-port of the template


def _template_text():
    return TEMPLATE.read_text(encoding="utf-8")


def _example_turn_secret():
    """The placeholder api/.env.example ships; the render must refuse it."""
    for line in (ROOT / "api/.env.example").read_text(encoding="utf-8").splitlines():
        if line.startswith("TURN_SECRET="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("api/.env.example no longer sets TURN_SECRET")


def test_template_denies_internal_peers_and_tcp_relay():
    lines = _template_text().splitlines()
    for needle in (
        "denied-peer-ip=0.0.0.0-0.255.255.255",
        "denied-peer-ip=10.0.0.0-10.255.255.255",
        "denied-peer-ip=100.64.0.0-100.127.255.255",
        "denied-peer-ip=127.0.0.0-127.255.255.255",
        "denied-peer-ip=169.254.0.0-169.254.255.255",
        "denied-peer-ip=172.16.0.0-172.31.255.255",
        "denied-peer-ip=192.168.0.0-192.168.255.255",
        "denied-peer-ip=::1",
        "denied-peer-ip=fc00::-fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
        "denied-peer-ip=fe80::-febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
        "no-tcp-relay",
    ):
        assert needle in lines, needle
    for prefix in ("user-quota=", "total-quota=", "max-bps="):
        assert any(line.startswith(prefix) for line in lines), prefix


def test_total_quota_matches_the_relay_port_range():
    # One relay port per allocation: a quota above the range can never be reached, below it wastes ports.
    t = _template_text()
    assert (
        f"min-port={RELAY_PORTS.start}" in t and f"max-port={RELAY_PORTS.stop - 1}" in t
    )
    assert f"total-quota={len(RELAY_PORTS)}" in t.splitlines()


def test_turn_ttl_default_is_short():
    # A subprocess reads the default without reloading api.constants, which would leak into other tests.
    env = {k: v for k, v in os.environ.items() if k != "TURN_CREDENTIAL_TTL"}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import api.constants as c; print(c.TURN_CREDENTIAL_TTL)",
        ],
        cwd=ROOT,
        env={**env, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        check=False,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip().splitlines()[-1]) <= 300


def test_no_static_turn_credentials_fallback(monkeypatch):
    from api.routes import turn_credentials, webrtc_signaling

    # get_ice_servers returns before the TURN code when these are off, so set them: otherwise this test is vacuous.
    monkeypatch.setattr(webrtc_signaling, "ENABLE_COTURN", True)
    monkeypatch.setattr(webrtc_signaling, "TURN_HOST", "turn.example.test")
    monkeypatch.setenv("TURN_USERNAME", "static")
    monkeypatch.setenv("TURN_PASSWORD", "static")

    # Control: with a secret the same path does hand out a temporary credential.
    key = secrets.token_hex(32)
    monkeypatch.setattr(webrtc_signaling, "TURN_SECRET", key)
    monkeypatch.setattr(turn_credentials, "TURN_SECRET", key)
    assert any(
        s.username and s.username.endswith(":u1")
        for s in webrtc_signaling.get_ice_servers(user_id="u1")
    )

    monkeypatch.setattr(webrtc_signaling, "TURN_SECRET", None)
    monkeypatch.setattr(turn_credentials, "TURN_SECRET", None)
    servers = webrtc_signaling.get_ice_servers(user_id="u1")
    assert servers, "STUN must remain"
    assert all(s.username is None for s in servers)


linux_only = pytest.mark.skipif(
    sys.platform == "win32", reason="GATED linux: the shell scripts are LF-only"
)


def _render(turn_secret, tmp_path):
    """Run dograh_render_remote_turn_conf as dograh-init does, with TURN_SECRET = `turn_secret` (None = unset)."""
    out = tmp_path / "turnserver.conf"
    env = {
        **os.environ,
        "TURN_EXTERNAL_IP": "203.0.113.7",
        "DOGRAH_DEPLOY_REPO_ROOT": str(ROOT),
    }
    env.pop("TURN_SECRET", None)
    if turn_secret is not None:
        env["TURN_SECRET"] = turn_secret
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'. "{SETUP_COMMON}"; dograh_render_remote_turn_conf "{tmp_path}" "{out}"',
        ],
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    return result, out


@linux_only
@pytest.mark.parametrize("which", ["example", "empty", "unset"])
def test_render_refuses_a_default_or_missing_turn_secret(which, tmp_path):
    result, out = _render(
        {"example": _example_turn_secret(), "empty": "", "unset": None}[which], tmp_path
    )
    assert result.returncode != 0
    for key in ("code=turn_secret_default", "where=", "reason=", "hint="):
        assert key in result.stderr, result.stderr
    assert "openssl rand -hex 32" in result.stderr
    assert (
        not out.exists()
    )  # a refused render must not leave a config behind for coturn to load


@linux_only
def test_render_accepts_a_real_turn_secret(tmp_path):
    key = secrets.token_hex(32)
    result, out = _render(key, tmp_path)
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert f"static-auth-secret={key}" in text
    assert "no-tcp-relay" in text
