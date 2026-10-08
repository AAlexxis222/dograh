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
BEGIN_DENY = "# BEGIN internal-peer deny"
END_DENY = "# END internal-peer deny"
RELAY_PORTS = range(49152, 49201)  # min-port / max-port of the template
DENIED_PEER_LINES = (
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
    # coturn compares address families strictly: the IPv4 ranges above do not cover these.
    "denied-peer-ip=::ffff:0.0.0.0-::ffff:255.255.255.255",
    "denied-peer-ip=64:ff9b::-64:ff9b::ffff:ffff",
)


def _template_text():
    return TEMPLATE.read_text(encoding="utf-8")


def _example_turn_secret():
    """The placeholder api/.env.example ships; the render must refuse it."""
    for line in (ROOT / "api/.env.example").read_text(encoding="utf-8").splitlines():
        if line.startswith("TURN_SECRET="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("api/.env.example no longer sets TURN_SECRET")


def _line_set(path):
    return set(path.read_text(encoding="utf-8").splitlines())


def test_template_denies_internal_ranges_between_markers():
    # The deny lines stay in the main template (the only file every install path fetches): an install that renders
    # with an older setup_common.sh then still denies. The markers let the render drop them for a private relay.
    lines = _template_text().splitlines()
    begin = next(i for i, line in enumerate(lines) if line.startswith(BEGIN_DENY))
    end = next(i for i, line in enumerate(lines) if line.startswith(END_DENY))
    assert begin < end
    assert set(DENIED_PEER_LINES) <= set(lines[begin:end])
    assert [line for line in lines if line.startswith("denied-peer-ip")] == [
        line for line in lines[begin:end] if line.startswith("denied-peer-ip")
    ]  # no deny line outside the block


def test_template_keeps_the_unconditional_hardening():
    lines = _line_set(TEMPLATE)
    assert "no-tcp-relay" in lines
    for prefix in ("total-quota=", "max-bps="):
        assert any(line.startswith(prefix) for line in lines), prefix
    # coturn counts user-quota per bare user id (it strips the timestamp), and embed calls all use the token
    # creator's id, so a per-user quota would cap concurrent calls per creator. Per-call quota waits for B3.
    assert not [line for line in lines if line.startswith("user-quota")]


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


def _render(turn_secret, tmp_path, external_ip="203.0.113.7", deny=None):
    """Run dograh_render_remote_turn_conf as dograh-init does.

    turn_secret None = unset; deny None = TURN_DENY_INTERNAL_PEERS unset."""
    out = tmp_path / "turnserver.conf"
    env = {
        **os.environ,
        "TURN_EXTERNAL_IP": external_ip,
        "DOGRAH_DEPLOY_PROJECT_DIR": str(
            tmp_path
        ),  # templates here win over the repo ones
        "DOGRAH_DEPLOY_REPO_ROOT": str(ROOT),
    }
    env.pop("TURN_SECRET", None)
    env.pop("TURN_DENY_INTERNAL_PEERS", None)
    if turn_secret is not None:
        env["TURN_SECRET"] = turn_secret
    if deny is not None:
        env["TURN_DENY_INTERNAL_PEERS"] = deny
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


def _rendered_lines(tmp_path, **kwargs):
    result, out = _render(secrets.token_hex(32), tmp_path, **kwargs)
    assert result.returncode == 0, result.stderr
    return out.read_text().splitlines()


@linux_only
@pytest.mark.parametrize(
    "external_ip",
    ["203.0.113.7", "8.8.8.8", "172.32.0.1", "100.128.0.1", "turn.example.test"],
)
def test_render_denies_internal_peers_when_the_relay_is_public(external_ip, tmp_path):
    lines = _rendered_lines(tmp_path, external_ip=external_ip)
    assert set(DENIED_PEER_LINES) <= set(lines)
    assert not [line for line in lines if line.startswith(("# BEGIN", "# END"))]


@linux_only
@pytest.mark.parametrize(
    "external_ip",
    [
        "192.168.1.10",
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.9",
        "172.31.255.1",
        "169.254.1.1",
        "100.64.0.1",
        "100.127.255.1",
    ],
)
def test_render_keeps_internal_peers_for_a_private_relay(external_ip, tmp_path):
    # A relay on a private address (local-turn, LAN, Tailscale) needs private peers or every TURN path fails.
    lines = _rendered_lines(tmp_path, external_ip=external_ip)
    assert not [line for line in lines if line.startswith("denied-peer-ip")]
    assert not [line for line in lines if line.startswith(("# BEGIN", "# END"))]
    assert "no-tcp-relay" in lines  # the rest of the hardening is unconditional


@linux_only
def test_deny_variable_forces_the_block_on_a_private_relay(tmp_path):
    lines = _rendered_lines(tmp_path, external_ip="192.168.1.10", deny="true")
    assert set(DENIED_PEER_LINES) <= set(lines)
    assert not [line for line in lines if line.startswith(("# BEGIN", "# END"))]


@linux_only
def test_deny_variable_can_switch_the_block_off_on_a_public_relay(tmp_path):
    lines = _rendered_lines(tmp_path, external_ip="203.0.113.7", deny="false")
    assert not [line for line in lines if line.startswith("denied-peer-ip")]
    assert not [line for line in lines if line.startswith(("# BEGIN", "# END"))]


@linux_only
def test_render_refuses_an_unknown_deny_value(tmp_path):
    result, out = _render(secrets.token_hex(32), tmp_path, deny="maybe")
    assert result.returncode != 0
    for key in ("code=turn_deny_internal_peers_invalid", "where=", "reason=", "hint="):
        assert key in result.stderr, result.stderr
    assert not out.exists()


@linux_only
@pytest.mark.parametrize("deny", ["true", "false"])
def test_render_leaves_a_template_without_markers_unchanged(deny, tmp_path):
    # dograh_template_path prefers <project>/deploy/templates, so a custom template can stand in for the shipped one.
    templates = tmp_path / "deploy/templates"
    templates.mkdir(parents=True)
    lines = [
        "listening-port=3478",
        "external-ip=__DOGRAH_TURN_EXTERNAL_IP__",
        "static-auth-secret=__DOGRAH_TURN_SECRET__",
    ]
    (templates / "turnserver.remote.conf.template").write_text("\n".join(lines) + "\n")
    key = secrets.token_hex(32)
    result, out = _render(key, tmp_path, external_ip="203.0.113.7", deny=deny)
    assert result.returncode == 0, result.stderr
    assert out.read_text().splitlines() == [
        "listening-port=3478",
        "external-ip=203.0.113.7",
        f"static-auth-secret={key}",
    ]
