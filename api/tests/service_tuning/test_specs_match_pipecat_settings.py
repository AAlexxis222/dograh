"""Every setting a SPECS row allows must exist in the pipecat Settings it targets.

``_fields`` reads the allow-list off the dataclass, so a bump cannot orphan a
name it derived. The names written by hand can: the excludes (guarded at import
by ``_fields``), the ``nullable_extra`` / ``settings_types`` / ``settings_choices``
/ ``settings_models`` keys, and any ``settings_allowed`` built from a literal. A
name pipecat no longer declares is either a 422-less no-op or a crash at run
creation, so each pipecat bump must make this test fail until the row is
re-derived. The provider-turn tables (``_PROVIDER_TURN_FIELDS`` /
``_PROVIDER_TURN_KNOBS``) are written by hand too and are the names a bump
orphans first, so they are held to the same rule.

Rows with no Settings class behind them (the ``realtime`` rows, ``tts._all``)
are not covered here: realtime is held in step with the factory by
test_realtime.py, and ``tts._all`` allows no setting.
"""

import dataclasses

import pytest

from api.services.pipecat.service_tuning_specs import (
    _PROVIDER_TURN_FIELDS,
    _PROVIDER_TURN_KNOBS,
    SPECS,
)

_ROWS = {key: spec for key, spec in SPECS.items() if spec.settings_classes()}


def _declared_fields(spec) -> set[str]:
    return {
        f.name for cls in spec.settings_classes() for f in dataclasses.fields(cls)
    }


def _named_by_hand(spec) -> set[str]:
    return (
        set(spec.settings_allowed)
        | spec.nullable_extra
        | spec.non_nullable
        | set(spec.settings_types)
        | set(spec.settings_choices)
        | set(spec.settings_models)
    )


@pytest.mark.parametrize("key,spec", sorted(_ROWS.items()))
def test_every_spec_field_exists_in_pipecat_settings(key, spec):
    classes = [cls.__name__ for cls in spec.settings_classes()]
    missing = _named_by_hand(spec) - _declared_fields(spec)
    assert not missing, f"{key}: not in {classes} (pipecat 1.12): {sorted(missing)}"


_TURN_SETTINGS = sorted(
    {("stt", p, n) for (p, _), names in _PROVIDER_TURN_FIELDS.items() for n in names}
    | {
        (kind, p, n)
        for kind, p, section, n in _PROVIDER_TURN_KNOBS
        if section == "settings"
    }
)
_TURN_CTOR = sorted(
    (kind, p, n) for kind, p, section, n in _PROVIDER_TURN_KNOBS if section == "ctor"
)


@pytest.mark.parametrize("kind,provider,name", _TURN_SETTINGS)
def test_every_provider_turn_setting_exists_in_pipecat_settings(kind, provider, name):
    spec = SPECS[(kind, provider)]
    assert name in _declared_fields(spec), (
        f"{kind}.{provider}: turn table names {name!r}, not in "
        f"{[c.__name__ for c in spec.settings_classes()]} (pipecat 1.12)"
    )


@pytest.mark.parametrize("kind,provider,name", _TURN_CTOR)
def test_every_provider_turn_ctor_kwarg_is_allowed_by_its_row(kind, provider, name):
    assert name in SPECS[(kind, provider)].ctor_allowed
