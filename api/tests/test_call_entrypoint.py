"""VOZ-N0-08: the `call` role drains live calls before uvicorn ever receives TERM; cell roles refuse a stale schema."""

import json
import os
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/xpand/call_entrypoint.sh"
REQUIRE_HEAD = ROOT / "scripts/xpand/require_db_head.sh"
START_SCRIPT = ROOT / "scripts/start_services_docker.sh"
DRAIN_SCRIPT = ROOT / "scripts/drain_web.sh"

posix_only = pytest.mark.skipif(os.name == "nt", reason="GATED bash_missing: POSIX signals; runs in CI")

# Stands in for `alembic -c api/alembic.ini <current|heads>`, with the output shapes captured from the real tool
# (fork head d7e3a915c2b8, real Postgres): two INFO lines, then one revision per line, "(head)" only on a head
# (a behind database prints the bare id). FAKE_AHEAD: the database holds a revision this tree lacks, so `current`
# prints alembic's own error and fails. FAKE_CURRENT_ERROR: `current` fails for any other reason. Records every call.
FAKE_ALEMBIC = """\
echo "$@" >> "$FAKE_ALEMBIC_LOG"
echo "INFO  [alembic.runtime.migration] Context impl PostgresqlImpl."
echo "INFO  [alembic.runtime.migration] Will assume transactional DDL."
case "${@: -1}" in
  current)
    if [[ -n "${FAKE_AHEAD:-}" ]]; then
      echo "ERROR [alembic.util.messaging] Can't locate revision identified by '$FAKE_AHEAD'"
      echo "FAILED: Can't locate revision identified by '$FAKE_AHEAD'"
      exit 255
    fi
    if [[ -n "${FAKE_CURRENT_ERROR:-}" ]]; then echo "$FAKE_CURRENT_ERROR"; exit 1; fi
    for r in $FAKE_CURRENT; do
      if [[ " $FAKE_HEADS " == *" $r "* ]]; then echo "$r (head)"; else echo "$r"; fi
    done ;;
  heads) for r in $FAKE_HEADS; do echo "$r (head)"; done ;;
esac
"""


@pytest.fixture
def alembic_env(tmp_path):
    """Env that points the scripts at a fake alembic whose DB is at `current` and whose code has `heads`."""
    fake = tmp_path / "fake_alembic.sh"
    fake.write_text(FAKE_ALEMBIC)
    log = tmp_path / "alembic.log"

    def make(current="abc123", heads="abc123", ahead="", current_error=""):
        return {
            **os.environ,
            "ALEMBIC_CMD": f"bash {fake}",
            "FAKE_ALEMBIC_LOG": str(log),
            "FAKE_CURRENT": current,
            "FAKE_HEADS": heads,
            "FAKE_AHEAD": ahead,
            "FAKE_CURRENT_ERROR": current_error,
        }

    make.log = log
    return make


# ---------------------------------------------------------------- call entrypoint: drain order and exit codes


@posix_only
def test_term_drains_before_stopping_uvicorn(tmp_path, alembic_env):
    log = tmp_path / "order.log"
    env = {
        **alembic_env(),
        "DRAIN_CMD": f"sleep 1; echo drained >> {log}",
        "CALL_CMD": f"python -c \"import signal,time,sys; signal.signal(signal.SIGTERM, lambda *a: (open(r'{log}','a').write('uvicorn_term\\n'), sys.exit(0))); time.sleep(60)\"",
    }
    p = subprocess.Popen(["bash", str(SCRIPT)], env=env)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) == 0
    assert log.read_text().split() == ["drained", "uvicorn_term"]


@posix_only
def test_failed_drain_still_stops_and_reports(alembic_env):
    env = {**alembic_env(), "DRAIN_CMD": "exit 7", "CALL_CMD": "sleep 60"}
    p = subprocess.Popen(["bash", str(SCRIPT)], env=env, stderr=subprocess.PIPE, text=True)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    # Exactly 1: the child's own 143 (killed by TERM) would also be non-zero, so != 0 would not prove the report.
    assert p.wait(timeout=10) == 1
    err = p.stderr.read()
    # VOZ-AC-B0-28: stable code plus where, reason and hint.
    for key in ("code=drain_failed", "where=", "reason=", "hint="):
        assert key in err, err


@posix_only
def test_repeated_term_drains_once_and_stops_uvicorn_once(tmp_path, alembic_env):
    # A second TERM during the drain must not re-run it; one during the child's slow stop must not re-signal it
    # (uvicorn's second TERM forces exit and skips the lifespan shutdown).
    log = tmp_path / "order.log"
    child = tmp_path / "child.py"
    child.write_text(
        "import signal, sys, time\n"
        "def stop(*a):\n"
        f"    open(r'{log}', 'a').write('uvicorn_term\\n')\n"
        "    time.sleep(1.5)\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "time.sleep(60)\n"
    )
    env = {
        **alembic_env(),
        "DRAIN_CMD": f"sleep 1; echo drained >> {log}",
        "CALL_CMD": f"exec python {child}",
    }
    p = subprocess.Popen(["bash", str(SCRIPT)], env=env)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)  # starts the drain
    time.sleep(0.3)
    p.send_signal(signal.SIGTERM)  # during the drain
    time.sleep(1.3)  # the drain ended at ~1.5s: the child is now in its slow stop
    p.send_signal(signal.SIGTERM)  # while waiting for the child
    assert p.wait(timeout=10) == 0
    assert log.read_text().split() == ["drained", "uvicorn_term"]


@posix_only
def test_call_serves_on_web_port_the_same_port_the_drain_polls(tmp_path, alembic_env):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    args = tmp_path / "uvicorn.args"
    fake = bindir / "uvicorn"
    fake.write_text(f'#!/usr/bin/env bash\necho "$@" > {args}\nexec sleep 60\n')
    fake.chmod(0o755)
    env = {
        **alembic_env(),
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "WEB_PORT": "8123",
        "UVICORN_PORT": "9999",  # the old, divergent knob: it must not decide the port
        "DRAIN_CMD": "true",
    }
    p = subprocess.Popen(["bash", str(SCRIPT)], env=env)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=10)
    assert "--port 8123" in args.read_text()


# ---------------------------------------------------------------- B5-44: a role on a stale schema refuses to start


@posix_only
def test_require_head_passes_when_db_is_at_head(alembic_env):
    r = subprocess.run(["bash", str(REQUIRE_HEAD), "call"], env=alembic_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@posix_only
def test_require_head_fails_with_a_named_message_when_behind(alembic_env):
    r = subprocess.run(["bash", str(REQUIRE_HEAD), "call"], env=alembic_env(current="old000"), capture_output=True, text=True)
    assert r.returncode != 0
    # VOZ-AC-B0-28 one-line shape.
    for key in ("code=db_schema_behind", "where=call", "reason=", "hint=", "cellctl migrate"):
        assert key in r.stderr, r.stderr
    assert len(r.stderr.strip().splitlines()) == 1
    # Without the overlay file compose starts the upstream image, whose tree lacks the fork head.
    assert "docker compose -f docker-compose.yaml -f docker-compose.cell.yaml run --rm api ./scripts/run_migrate.sh" in r.stderr


@posix_only
def test_require_head_fails_on_an_unmigrated_db(alembic_env):
    r = subprocess.run(["bash", str(REQUIRE_HEAD), "call"], env=alembic_env(current=""), capture_output=True, text=True)
    assert r.returncode != 0 and "code=db_schema_behind" in r.stderr


@posix_only
def test_require_head_fails_on_multiple_heads(alembic_env):
    r = subprocess.run(
        ["bash", str(REQUIRE_HEAD), "call"], env=alembic_env(current="aaa111", heads="aaa111 bbb222"), capture_output=True, text=True
    )
    assert r.returncode != 0 and "code=db_schema_behind" in r.stderr


@posix_only
def test_require_head_starts_with_a_named_warning_when_db_is_ahead(alembic_env):
    # B5-43/45: an old image restarted after a migration, or a rolled-back image, sees a revision it does not know.
    r = subprocess.run(["bash", str(REQUIRE_HEAD), "call"], env=alembic_env(ahead="ffffffffffff"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # VOZ-AC-B0-28 one-line shape, every field non-empty.
    assert re.fullmatch(r"code=db_schema_ahead where=call reason=\S.* hint=\S.*", r.stderr.strip()), r.stderr
    assert "ffffffffffff" in r.stderr


@posix_only
@pytest.mark.parametrize(
    "current_error, alembic_cmd",
    [("ConnectionRefusedError: [Errno 111] Connect call failed", None), ("", "false")],
    ids=["current_fails_database_unreachable", "alembic_fails_before_reading"],
)
def test_require_head_fails_when_alembic_itself_fails(alembic_env, current_error, alembic_cmd):
    env = alembic_env(current_error=current_error)
    if alembic_cmd:
        env["ALEMBIC_CMD"] = alembic_cmd
    r = subprocess.run(["bash", str(REQUIRE_HEAD), "call"], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "code=db_schema_unreadable" in r.stderr
    hint = r.stderr.split("hint=", 1)[1]
    assert "DATABASE_URL" in hint and "migrat" not in hint, hint  # about reaching the database, not about migrating


@posix_only
def test_require_head_refuses_incoherent_durations_before_touching_the_database(alembic_env):
    # VOZ-AC-B3-55: every role asserts the cell durations at startup; the failure is named and nothing else runs.
    env = {**alembic_env(), "CELL_CALL_DURATION_CEILING_S": "0"}
    r = subprocess.run(["bash", str(REQUIRE_HEAD), "call"], env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert re.search(r"code=knob_out_of_range where=call reason=\S.* hint=\S.*", r.stderr), r.stderr
    assert not alembic_env.log.exists()


def _start_script_env(base):
    # Every optional duty off: only the migration gate of the start script is under test.
    return {**base, "RUN_MIGRATIONS_ON_START": "false", "ENABLE_ARI_MANAGER": "false",
            "ENABLE_CAMPAIGN_ORCHESTRATOR": "false", "FASTAPI_WORKERS": "0", "ARQ_WORKERS": "0"}


@posix_only
def test_start_script_never_upgrades_and_refuses_a_stale_schema(alembic_env):
    r = subprocess.run(
        ["bash", str(START_SCRIPT)], env=_start_script_env(alembic_env(current="old000")), capture_output=True, text=True
    )
    assert r.returncode != 0 and "code=db_schema_behind" in r.stderr, r.stderr
    assert "upgrade" not in alembic_env.log.read_text()


@posix_only
def test_start_script_runs_when_db_is_at_head_and_still_never_upgrades(alembic_env):
    r = subprocess.run(["bash", str(START_SCRIPT)], env=_start_script_env(alembic_env()), capture_output=True, text=True)
    assert "db_schema_behind" not in r.stderr, r.stderr
    assert "ari_manager disabled" in r.stdout  # got past the gate to the service section
    assert "upgrade" not in alembic_env.log.read_text()


@posix_only
def test_call_entrypoint_refuses_a_stale_schema_before_launching_uvicorn(tmp_path, alembic_env):
    started = tmp_path / "started"
    env = {**alembic_env(current="old000"), "CALL_CMD": f"touch {started}; sleep 60", "DRAIN_CMD": "true"}
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10)
    assert r.returncode != 0 and "code=db_schema_behind" in r.stderr, r.stderr
    assert not started.exists()


# ---------------------------------------------------------------- drain_web.sh fail-closed


class _ActiveCalls(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"active_calls": 2}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def busy_server():
    server = HTTPServer(("127.0.0.1", 0), _ActiveCalls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def _drain(port, fail_closed):
    env = {**os.environ, "DOGRAH_DEVOPS_SECRET": "x", "WEB_PORT": str(port), "DRAIN_INITIAL_DELAY": "0",
           "DRAIN_INTERVAL": "1", "DRAIN_MAX_WAIT": "2"}
    if fail_closed:
        env["DRAIN_FAIL_CLOSED"] = "true"
    return subprocess.run(["sh", str(DRAIN_SCRIPT)], env=env, capture_output=True, text=True, timeout=20)


@posix_only
def test_drain_times_out_with_calls_still_active_fails_closed(busy_server):
    r = _drain(busy_server, fail_closed=True)
    assert r.returncode != 0
    for key in ("code=drain_timeout", "where=", "reason=", "hint="):
        assert key in r.stderr, r.stderr


@posix_only
def test_drain_default_still_proceeds_after_the_timeout(busy_server):
    assert _drain(busy_server, fail_closed=False).returncode == 0


@posix_only
def test_drain_never_reading_the_counter_fails_closed():
    r = _drain(1, fail_closed=True)  # nothing listens on port 1
    assert r.returncode != 0
    for key in ("code=drain_counter_unreadable", "where=", "reason=", "hint="):
        assert key in r.stderr, r.stderr


@posix_only
def test_drain_default_proceeds_when_the_counter_is_unreadable():
    assert _drain(1, fail_closed=False).returncode == 0


@posix_only
def test_drain_flag_is_cleared_at_start_and_set_on_term(tmp_path, alembic_env):
    # active-calls reports draining = this file exists: stale at start would report draining forever.
    flag = tmp_path / "draining"
    flag.touch()
    seen = tmp_path / "seen.log"
    env = {
        **alembic_env(),
        "DRAIN_FLAG_FILE": str(flag),
        "DRAIN_CMD": f"test -e {flag} && echo flag_set_before_drain > {seen}",
        "CALL_CMD": "sleep 60",
    }
    p = subprocess.Popen(["bash", str(SCRIPT)], env=env)
    time.sleep(0.5)
    assert not flag.exists()
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=10)
    assert seen.read_text().strip() == "flag_set_before_drain"
    assert flag.exists()


@posix_only
def test_require_db_head_runs_the_cell_startup_check_before_alembic(alembic_env):
    # arq and coordinators never run the api lifespan: this gate is where they refuse DEBUG logs (VOZ-AC-B6-52).
    env = {**alembic_env(), "CELL_ROLE": "arq", "DOGRAH_DEVOPS_SECRET": "s", "LOG_LEVEL": "DEBUG"}
    r = subprocess.run(["bash", str(REQUIRE_HEAD), "arq"], env=env, capture_output=True, text=True)
    assert r.returncode == 1
    for key in ("code=log_level_debug_in_cell", "where=arq", "reason=", "hint="):
        assert key in r.stderr, r.stderr
    assert not alembic_env.log.exists()
