"""Explicit ``null`` per service (spec §5.2, #16).

For every row of SPECS: each nullable setting takes ``null`` (a 200 that
stores the tri-state "send this field unset"), and one non-nullable setting
of the row rejects it by name. The gated settings (provider turn detection,
``_PROVIDER_TURN_FIELDS`` / ``_PROVIDER_TURN_KNOBS``) are left out: their
422 names the gate, whatever the value.
"""

import pytest
from pydantic import ValidationError

from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.pipecat import service_tuning_specs as specs

_GATED = {
    (kind, provider, name)
    for kind, provider, section, name in (
        set(specs._PROVIDER_TURN_FIELDS) | set(specs._PROVIDER_TURN_KNOBS)
    )
    if section == "settings"
}


def _doc(kind, provider, name):
    tuning = {"settings": {name: None}}
    if (kind, provider) == ("stt", "openai") and name not in specs._fields(
        specs.OpenAISTTSettings
    ):
        # A realtime-only field is a 422 under the default segments api.
        tuning["options"] = {"api": "realtime"}
    return {"service_tuning": {kind: {provider: tuning}}}


def _cases():
    nullable, non_nullable = [], []
    for (kind, provider), spec in sorted(specs.SPECS.items()):
        ok = [
            n
            for n in sorted(spec.settings_allowed)
            if (kind, provider, n) not in _GATED
        ]
        for name in ok:
            if name in spec.nullable():
                nullable.append((kind, provider, name))
        first = next((n for n in ok if n not in spec.nullable()), None)
        if first is not None:
            non_nullable.append((kind, provider, first))
    return nullable, non_nullable


NULLABLE, NON_NULLABLE = _cases()


def _id(case):
    return ".".join(case)


@pytest.mark.parametrize("kind,provider,name", NULLABLE, ids=map(_id, NULLABLE))
def test_null_is_accepted_on_every_nullable_setting(kind, provider, name):
    cfg = WorkflowConfigurationDefaults.model_validate(_doc(kind, provider, name))
    stored = cfg.model_dump(exclude_unset=True)["service_tuning"][kind][provider]
    assert stored["settings"] == {name: None}  # tri-state survives


@pytest.mark.parametrize("kind,provider,name", NON_NULLABLE, ids=map(_id, NON_NULLABLE))
def test_null_is_a_named_422_on_a_non_nullable_setting(kind, provider, name):
    with pytest.raises(
        ValidationError,
        match=rf"{kind}\.{provider}\.settings\.{name}: null not allowed",
    ):
        WorkflowConfigurationDefaults.model_validate(_doc(kind, provider, name))


def test_matrix_reaches_every_row_that_declares_a_setting():
    rows_with_settings = {
        (k, p) for (k, p), s in specs.SPECS.items() if s.settings_allowed
    }
    covered = {(k, p) for k, p, _ in NULLABLE} | {(k, p) for k, p, _ in NON_NULLABLE}
    assert covered == rows_with_settings
    # The realtime rows are all non-nullable by design (specs module comment).
    assert not [c for c in NULLABLE if c[0] == "realtime"]
