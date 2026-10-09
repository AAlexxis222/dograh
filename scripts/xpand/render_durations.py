"""Renders every deployment number that derives from the cell call-duration ceiling (VOZ-AC-B3-56).

The cell env (CELL_CALL_DURATION_CEILING_S and CELL_STOP_GRACE_S, read by the Compose overlay) is written from the
env output; the Helm values and drain scripts carry the values rendered for the default ceiling, and
api/tests/test_durations.py fails when one of them is edited by hand. A cell with another ceiling re-renders:
`python -m scripts.xpand.render_durations` prints the values for CELL_CALL_DURATION_CEILING_S. Each role then checks
at startup that the grace and drain values it runs with are not below the ones derived from its ceiling.
"""

import json

from api.services.runtime.durations import CEILING_ENV, STOP_GRACE_ENV, Durations


def render(d: Durations) -> dict:
    return {
        "compose": {"services": {"call": {"stop_grace_period": f"{d.grace}s"}}},
        "helm": {
            "web": {
                "terminationGracePeriodSeconds": d.grace,
                "preStopSleepSeconds": d.pre_stop_delay_s,
                "drainMaxWaitSeconds": d.drain_max,
            }
        },
        "env": {
            CEILING_ENV: str(d.ceiling),
            STOP_GRACE_ENV: str(d.grace),
            "DRAIN_TIMEOUT": str(d.drain_max),
            "DRAIN_MAX_WAIT": str(d.drain_max),
            "DRAIN_INITIAL_DELAY": str(d.pre_stop_delay_s),
        },
    }


if __name__ == "__main__":
    print(json.dumps(render(Durations.from_env()), indent=2))
