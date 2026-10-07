"""VOZ-N0-08: the `call` role drains live calls before uvicorn ever receives TERM."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/xpand/call_entrypoint.sh"


@pytest.mark.skipif(os.name == "nt", reason="GATED bash_missing: POSIX signals; runs in CI")
def test_term_drains_before_stopping_uvicorn(tmp_path):
    log = tmp_path / "order.log"
    env = {
        **os.environ,
        "DRAIN_CMD": f"sleep 1; echo drained >> {log}",
        "CALL_CMD": f"python -c \"import signal,time,sys; signal.signal(signal.SIGTERM, lambda *a: (open(r'{log}','a').write('uvicorn_term\\n'), sys.exit(0))); time.sleep(60)\"",
    }
    p = subprocess.Popen(["bash", str(SCRIPT)], env=env)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) == 0
    assert log.read_text().split() == ["drained", "uvicorn_term"]


@pytest.mark.skipif(os.name == "nt", reason="GATED bash_missing")
def test_failed_drain_still_stops_and_reports(tmp_path):
    env = {**os.environ, "DRAIN_CMD": "exit 7", "CALL_CMD": "sleep 60"}
    p = subprocess.Popen(["bash", str(SCRIPT)], env=env, stderr=subprocess.PIPE, text=True)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) != 0
    err = p.stderr.read()
    # VOZ-AC-B0-28: stable code plus where, reason and hint.
    for key in ("code=drain_failed", "where=", "reason=", "hint="):
        assert key in err, err
