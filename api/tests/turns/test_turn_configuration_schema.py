"""`turn` as workflow configuration: bounded knobs, absent by default, clamped with a
warning when a stored document is out of range (spec §2.2, §6.1.1)."""

import pytest
from pydantic import ValidationError

from api.schemas.turn_configuration import (
    DEFAULT_HYBRID_HOLD_MS,
    DEFAULT_HYBRID_WAIT_MS,
    TurnConfiguration,
)
from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.configuration.cascade import resolve_effective_workflow_configurations


def test_turn_absent_by_default_does_not_materialise():
    doc = WorkflowConfigurationDefaults().model_dump(exclude_none=True)
    assert "turn" not in doc


def test_turn_defaults():
    turn = TurnConfiguration()
    assert turn.source == "auto"
    assert turn.hybrid.wait_ms == DEFAULT_HYBRID_WAIT_MS == 0
    assert turn.hybrid.hold_ms == DEFAULT_HYBRID_HOLD_MS == 1500


@pytest.mark.parametrize(
    "payload",
    [
        {"source": "flux"},
        {"hybrid": {"wait_ms": -1}},
        {"hybrid": {"wait_ms": 2001}},
        {"hybrid": {"hold_ms": 10001}},
        {"hybrid": {"bogus": 1}},
    ],
)
def test_turn_rejects_unknown_hybrid_keys_and_out_of_range(payload):
    with pytest.raises(ValidationError):
        WorkflowConfigurationDefaults(turn=payload)


def test_turn_keeps_the_reserved_sibling_namespaces():
    # `turn.ignore_terms` (cascade merge rule) and `turn.analyzer.url` (secrets
    # registry) belong to later parts: a document carrying them must still validate
    # and survive a round trip, like every other section of the document.
    doc = WorkflowConfigurationDefaults(
        turn={"source": "local", "ignore_terms": {"terms": ["ok"]}}
    ).model_dump(exclude_unset=True)
    assert doc["turn"]["ignore_terms"] == {"terms": ["ok"]}


def test_turn_round_trips_through_defaults_model():
    wf = WorkflowConfigurationDefaults(
        turn={"source": "local", "hybrid": {"hold_ms": 800}}
    )
    doc = wf.model_dump(exclude_none=True)
    assert doc["turn"] == {"source": "local", "hybrid": {"wait_ms": 0, "hold_ms": 800}}


def test_resolver_clamps_stored_hybrid_knobs_with_warning():
    # Stored documents bypass the PUT schema; the resolver clamps and warns (spec §2.2).
    resolved = resolve_effective_workflow_configurations(
        organization_defaults={},
        definition_configurations={
            "turn": {"source": "local", "hybrid": {"wait_ms": 5000, "hold_ms": -5}}
        },
    )
    assert resolved.effective["turn"]["hybrid"]["wait_ms"] == 2000
    assert resolved.effective["turn"]["hybrid"]["hold_ms"] == 0
    assert any("turn.hybrid.wait_ms" in w for w in resolved.warnings)
    assert any("turn.hybrid.hold_ms" in w for w in resolved.warnings)
