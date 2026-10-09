"""VOZ-N0-20 / VOZ-AT-B3-19: durations.py is the single source of call, slot, drain and grace durations."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from api.services.call_concurrency.rate_limiter import RateLimiter
from api.services.runtime.durations import (
    CEILING_ENV,
    DEFAULT_MAX_CALL_DURATION_S,
    STOP_GRACE_ENV,
    Durations,
    DurationsError,
    cell_durations,
    main,
)
from api.tests.support.cell_overlay import overlay_source

ROOT = Path(__file__).resolve().parents[2]


def test_default_ceiling_derivations():
    d = Durations.from_cell(ceiling_s=1200)
    assert (d.slot_ttl, d.drain_max, d.grace) == (1260, 1260, 1305)
    assert d.heartbeat_renew_s <= d.slot_ttl / 3


def test_raised_ceiling_keeps_invariants():
    d = Durations.from_cell(ceiling_s=7200)
    assert d.ceiling < d.slot_ttl <= d.drain_max < d.grace


def test_workflow_max_above_ceiling_is_named_error():
    with pytest.raises(DurationsError) as e:
        Durations.from_cell(ceiling_s=1200).check_workflow_max(1800)
    assert (
        e.value.code == "knob_out_of_range"
        and "cell.call_duration_ceiling_s" in e.value.hint
    )


def test_incoherent_margins_are_a_named_error():
    with pytest.raises(DurationsError) as e:
        Durations.from_cell(ceiling_s=1200, slot_margin_s=0)
    assert e.value.code == "durations_incoherent" and e.value.reason and e.value.hint
    # The margins are code constants today: the hint must point at what an operator can change, not at a file that does not exist.
    assert "durations.py" in e.value.hint and "yaml" not in e.value.hint


def test_ceiling_below_the_default_workflow_max_is_a_named_error():
    with pytest.raises(DurationsError) as e:
        Durations.from_cell(ceiling_s=DEFAULT_MAX_CALL_DURATION_S - 1)
    assert e.value.code == "knob_out_of_range" and CEILING_ENV in e.value.hint
    assert (
        f">= {DEFAULT_MAX_CALL_DURATION_S}" in e.value.hint
    )  # the hint states the real lower bound
    assert (
        Durations.from_cell(ceiling_s=DEFAULT_MAX_CALL_DURATION_S).ceiling
        == DEFAULT_MAX_CALL_DURATION_S
    )


@pytest.mark.parametrize("ceiling", [0, -5])
def test_non_positive_ceiling_is_a_named_error(ceiling):
    with pytest.raises(DurationsError) as e:
        Durations.from_cell(ceiling_s=ceiling)
    assert e.value.code == "knob_out_of_range" and CEILING_ENV in e.value.hint


def test_from_env_defaults_to_1200_and_reads_the_cell_knob():
    assert Durations.from_env({}).ceiling == 1200
    assert Durations.from_env({CEILING_ENV: "7200"}).grace == 15 + 7260 + 30


def test_from_env_rejects_a_non_integer_ceiling():
    with pytest.raises(DurationsError) as e:
        Durations.from_env({CEILING_ENV: "20m"})
    assert e.value.code == "knob_invalid" and CEILING_ENV in e.value.hint


def test_deployed_values_below_the_derived_ones_are_a_named_error():
    d = Durations.from_env({CEILING_ENV: "7200"})
    for name, derived in [
        (STOP_GRACE_ENV, d.grace),
        ("DRAIN_MAX_WAIT", d.drain_max),
        ("DRAIN_TIMEOUT", d.drain_max),
    ]:
        with pytest.raises(DurationsError) as e:
            d.check_deployed({name: str(derived - 1)})
        assert (
            e.value.code == "durations_incoherent"
            and name in e.value.reason
            and e.value.hint
        )
        d.check_deployed({name: str(derived)})  # equal is coherent
    d.check_deployed({})  # unset = not checked


def test_deployed_value_that_is_not_a_number_is_a_named_error():
    with pytest.raises(DurationsError) as e:
        Durations.from_cell(ceiling_s=1200).check_deployed({STOP_GRACE_ENV: "21m"})
    assert e.value.code == "knob_invalid" and STOP_GRACE_ENV in e.value.reason


def test_startup_assertion_compares_the_rendered_grace_with_the_ceiling(
    monkeypatch, capsys
):
    # The ceiling was raised but the grace of the cell env was not re-rendered: the role must refuse to start.
    monkeypatch.setenv(CEILING_ENV, "7200")
    monkeypatch.setenv(STOP_GRACE_ENV, str(Durations.from_cell(ceiling_s=1200).grace))
    assert main("call") == 1
    line = capsys.readouterr().err.strip()
    assert re.fullmatch(
        r"code=durations_incoherent where=call reason=\S.* hint=\S.*", line
    ), line
    monkeypatch.setenv(STOP_GRACE_ENV, str(Durations.from_cell(ceiling_s=7200).grace))
    assert main("call") == 0


def test_render_outputs_match(tmp_path):
    from scripts.xpand.render_durations import render

    out = render(Durations.from_cell(ceiling_s=7200))
    assert (
        out["compose"]["services"]["call"]["stop_grace_period"]
        == f"{Durations.from_cell(ceiling_s=7200).grace}s"
    )
    assert out["env"]["DRAIN_MAX_WAIT"] == str(
        Durations.from_cell(ceiling_s=7200).drain_max
    )


def _script_default(script: str, var: str) -> str:
    """The literal after `${VAR:-` in a script: the value that runs when nobody exports VAR."""
    found = re.search(
        r"\$\{" + var + r":-(\d+)\}",
        (ROOT / "scripts" / script).read_text(encoding="utf-8"),
    )
    assert found, f"{script} has no ${{{var}:-<seconds>}} default"
    return found.group(1)


def test_committed_files_carry_the_rendered_default_values():
    """VOZ-AT-B3-19: a hand edit of any rendered number fails here, so the files cannot drift from durations.py."""
    from scripts.xpand.render_durations import render

    d = Durations.from_cell(ceiling_s=1200)
    out = render(d)

    # The overlay carries no number for the call role: Compose reads the rendered cell env, and so does every role.
    overlay = overlay_source()
    assert (
        overlay["services"]["call"]["stop_grace_period"] == "${" + STOP_GRACE_ENV + "}s"
    )
    for name in (CEILING_ENV, STOP_GRACE_ENV):
        required = (
            "${"
            + name
            + ":?"
            + name
            + " must be set (render it: python -m scripts.xpand.render_durations)}"
        )
        assert overlay["x-cell-env"][name] == required
    reference = (
        (ROOT / "deploy/cell/.env.reference").read_text(encoding="utf-8").split()
    )
    assert {
        CEILING_ENV,
        STOP_GRACE_ENV,
        "DRAIN_MAX_WAIT",
        "DRAIN_INITIAL_DELAY",
    } <= set(reference)
    assert (out["env"][CEILING_ENV], out["env"][STOP_GRACE_ENV]) == (
        str(d.ceiling),
        str(d.grace),
    )

    helm_web = yaml.safe_load(
        (ROOT / "deploy/helm/dograh/values.yaml").read_text(encoding="utf-8")
    )["web"]
    for key, value in out["helm"]["web"].items():
        assert helm_web[key] == value, key
    helm_readme = (ROOT / "deploy/helm/dograh/README.md").read_text(encoding="utf-8")
    assert f"{d.grace}s" in helm_readme and f"{d.drain_max}s" in helm_readme

    assert (
        _script_default("rolling_update.sh", "DRAIN_TIMEOUT")
        == out["env"]["DRAIN_TIMEOUT"]
    )
    assert (
        _script_default("drain_web.sh", "DRAIN_MAX_WAIT")
        == out["env"]["DRAIN_MAX_WAIT"]
    )
    assert (
        _script_default("drain_web.sh", "DRAIN_INITIAL_DELAY")
        == out["env"]["DRAIN_INITIAL_DELAY"]
    )


def test_no_ttl_literals_left_in_call_concurrency():
    src = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (ROOT / "api/services/call_concurrency").glob("*.py")
    )
    assert not re.search(r"\b(1200|3600|1800)\b", src)


# Numeric TTL literals that stay in call_concurrency, each named: neither is a call duration.
ALLOWED_TTL_SITES = {
    "redis.call('EXPIRE', key, 2)": "acquire_token: the 2 s sliding window of the per-second rate limit",
    "redis.call('EXPIRE', KEYS[1], 86400)": "select_from_number: the daily caller-ID rotation counter",
}
TTL_CALL = re.compile(r"(?i)\b(?:expire|pexpire|setex)\b[^\n]*?\b\d+\b|\bex\s*=\s*\d+")


def test_every_ttl_in_call_concurrency_is_derived_or_a_named_exception():
    """Not only 1200/3600/1800: any numeric EXPIRE / expire / setex / ex= literal, so a hand-written 1260 fails too."""
    lines = [
        line.strip()
        for p in (ROOT / "api/services/call_concurrency").glob("*.py")
        for line in p.read_text(encoding="utf-8").splitlines()
    ]
    hits = [line for line in lines if TTL_CALL.search(line)]
    assert [
        line for line in hits if not any(site in line for site in ALLOWED_TTL_SITES)
    ] == []
    for site in ALLOWED_TTL_SITES:  # the allowlist cannot rot into a blanket pass
        assert any(site in line for line in lines), site


def test_slot_ttl_consumers_follow_a_raised_ceiling():
    """VOZ-AT-B3-19 at ceiling 7200, in a fresh interpreter: the constants are read at import."""
    code = (
        "import json;"
        "from api.schemas.workflow_configurations import MAX_CALL_DURATION_SECONDS as cap;"
        "from api.services.configuration.cascade import NUMERIC_BOUNDS;"
        "from api.services.call_concurrency.rate_limiter import RateLimiter;"
        "print(json.dumps({'cap': cap, 'stale': RateLimiter().stale_call_timeout,"
        " 'cascade': [b[2] for b in NUMERIC_BOUNDS if b[0] == ('max_call_duration',)][0]}))"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, CEILING_ENV: "7200"},
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    seen = json.loads(run.stdout.strip().splitlines()[-1])
    assert seen == {
        "cap": 7200,
        "cascade": 7200,
        "stale": Durations.from_cell(ceiling_s=7200).slot_ttl,
    }


async def test_acquire_gives_the_lua_script_the_slot_ttl():
    limiter = RateLimiter(Durations.from_cell(ceiling_s=7200))
    client = AsyncMock()
    client.eval.return_value = None
    limiter._get_redis = AsyncMock(return_value=client)
    await limiter.try_acquire_concurrent_slot_details(1, 5)
    d = Durations.from_cell(ceiling_s=7200)
    assert client.eval.await_args.args[-2:] == (d.pending_ttl_s, d.slot_ttl)
    await limiter.renew_slot(org_id=1, attempt_id="a", workflow_run_id=7)
    # (script, numkeys, clock, org, scope, fleet, mapping, legacy org, member, ttl, org_id, scope)
    assert client.eval.await_args.args[9] == d.slot_ttl


async def test_workflow_slot_mapping_ttl_is_the_slot_ttl():
    """Both mapping writers, at ceiling 7200: a TTL held in a constant (not a literal on the EXPIRE line) still fails here."""
    slot_ttl = Durations.from_cell(ceiling_s=7200).slot_ttl
    limiter = RateLimiter(Durations.from_cell(ceiling_s=7200))
    client = AsyncMock()
    limiter._get_redis = AsyncMock(return_value=client)
    await limiter.store_workflow_slot_mapping(7, 1, "slot")
    assert client.expire.await_args.args == ("workflow_slot_mapping:7", slot_ttl)
    await limiter.store_workflow_slot_mapping_if_absent(7, 1, "slot")
    assert (
        client.eval.await_args.args[5] == slot_ttl
    )  # (script, numkeys, key, org_id, slot_id, ttl, scope_key)


def test_direct_construction_is_validated_too():
    with pytest.raises(DurationsError) as e:
        Durations(ceiling=5)
    assert e.value.code == "knob_out_of_range"
    with pytest.raises(DurationsError) as e:
        Durations(ceiling=1200, slot_margin_s=0)
    assert e.value.code == "durations_incoherent"


def _rendered_env(d):
    return {name: str(value) for name, value in d.deployed_env().items()}


def test_render_and_the_gate_share_one_name_to_value_mapping():
    from scripts.xpand.render_durations import render

    for ceiling in (1200, 7200):
        d = Durations.from_cell(ceiling_s=ceiling)
        assert render(d)["env"] == _rendered_env(d)
        d.check_deployed(_rendered_env(d))  # an env as rendered is coherent


def test_deployed_initial_delay_that_overruns_the_grace_is_a_named_error():
    d = Durations.from_cell(ceiling_s=1200)
    env = {
        **_rendered_env(d),
        "DRAIN_INITIAL_DELAY": "600",
    }  # 600 + 1260 + 30 > 1305: SIGKILL mid-drain
    with pytest.raises(DurationsError) as e:
        d.check_deployed(env)
    assert (
        e.value.code == "durations_incoherent"
        and "DRAIN_INITIAL_DELAY" in e.value.reason
        and e.value.hint
    )
    # The SIGTERM headroom counts: 40 + 1260 fits the grace of 1305, 40 + 1260 + 30 does not.
    with pytest.raises(DurationsError):
        d.check_deployed({**_rendered_env(d), "DRAIN_INITIAL_DELAY": "40"})
    d.check_deployed({"DRAIN_INITIAL_DELAY": "600"})  # not all three set: not checked
    with pytest.raises(DurationsError) as e:
        d.check_deployed({"DRAIN_INITIAL_DELAY": "soon"})
    assert e.value.code == "knob_invalid"


def test_the_ceiling_is_resolved_once_for_every_consumer():
    from api.schemas import workflow_configurations as schema

    assert cell_durations() is cell_durations()
    assert schema.MAX_CALL_DURATION_SECONDS == cell_durations().ceiling
    assert RateLimiter().stale_call_timeout == cell_durations().slot_ttl


def test_the_schema_validator_asks_the_resolved_durations(monkeypatch):
    from api.schemas import workflow_configurations as schema

    monkeypatch.setattr(
        schema, "cell_durations", lambda: Durations.from_cell(ceiling_s=400)
    )
    with pytest.raises(Exception) as e:
        schema.WorkflowConfigurationDefaults(max_call_duration=500)
    assert "knob_out_of_range" in str(e.value)


def test_the_rate_limiter_asks_the_resolved_durations(monkeypatch):
    from api.services.call_concurrency import rate_limiter as module

    monkeypatch.setattr(
        module, "cell_durations", lambda: Durations.from_cell(ceiling_s=400)
    )
    assert (
        module.RateLimiter().stale_call_timeout
        == Durations.from_cell(ceiling_s=400).slot_ttl
    )


def test_ring_timeout_has_the_tsr_floor_and_each_carrier_gets_at_most_its_documented_max():
    with pytest.raises(DurationsError) as e:
        Durations(ceiling=1200, ring_timeout_s=14)
    assert (
        e.value.code == "knob_out_of_range" and "16 CFR 310.4(b)(4)" in e.value.reason
    )
    d = Durations(ceiling=1200, ring_timeout_s=300)
    assert d.ring_timeout_for("vonage") == 120  # ringing_timer max
    assert d.ring_timeout_for("twilio") == 300
    assert d.ring_timeout_for("ari") == 300  # no documented max: as configured
    assert d.outbound_pending_ttl_s == 300 + d.pending_ttl_s
