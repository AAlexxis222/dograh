"""Renders every deployment number that derives from the cell call-duration ceiling (VOZ-AC-B3-56).

The committed Compose overlay, Helm values and drain scripts carry the values rendered for the default ceiling;
api/tests/test_durations.py fails when one of them is edited by hand. A cell with another ceiling re-renders:
`python -m scripts.xpand.render_durations` prints the values for CELL_CALL_DURATION_CEILING_S.
"""

import json

from api.services.runtime.durations import CEILING_ENV, Durations


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
            "DRAIN_TIMEOUT": str(d.drain_max),
            "DRAIN_MAX_WAIT": str(d.drain_max),
        },
    }


if __name__ == "__main__":
    print(json.dumps(render(Durations.from_env()), indent=2))
