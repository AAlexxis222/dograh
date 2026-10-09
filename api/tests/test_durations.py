"""VOZ-N0-20 / VOZ-AT-B3-19: durations.py is the single source of call, slot, drain and grace durations."""

import re
from pathlib import Path

import pytest
import yaml

from api.services.runtime.durations import CEILING_ENV, Durations, DurationsError
from api.tests.test_cell_overlay import _overlay_source

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

    out = render(Durations.from_cell(ceiling_s=1200))
    overlay_call = _overlay_source()["services"]["call"]
    assert (
        overlay_call["stop_grace_period"]
        == out["compose"]["services"]["call"]["stop_grace_period"]
    )

    helm_web = yaml.safe_load(
        (ROOT / "deploy/helm/dograh/values.yaml").read_text(encoding="utf-8")
    )["web"]
    for key, value in out["helm"]["web"].items():
        assert helm_web[key] == value, key

    assert (
        _script_default("rolling_update.sh", "DRAIN_TIMEOUT")
        == out["env"]["DRAIN_TIMEOUT"]
    )
    assert (
        _script_default("drain_web.sh", "DRAIN_MAX_WAIT")
        == out["env"]["DRAIN_MAX_WAIT"]
    )


def test_no_ttl_literals_left_in_call_concurrency():
    src = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (ROOT / "api/services/call_concurrency").glob("*.py")
    )
    assert not re.search(r"\b(1200|3600|1800)\b", src)
