"""Outbound readiness across the provider registry.

Provider-specific rules live next to their provider; this covers the contract
every provider participates in — the connectivity split the UI routes off, and
the shared "you still need a number" fallback that keeps a credentials-only
configuration from reporting itself ready.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import api.services.telephony.providers  # noqa: F401  (registers every spec)
from api.services.telephony import registry
from api.services.telephony.factory import get_setup_checklist
from api.services.telephony.outbound_readiness import (
    OutboundSetupIncompleteError,
    resolve_outbound_configuration_id,
)


def _step(checklist, key):
    return next(step for step in checklist.steps if step.key == key)


def test_connectivity_separates_carrier_accounts_from_byo_sip():
    """The UI offers two routes off this field, so the split must hold."""
    by_connectivity: dict[str, set[str]] = {"api": set(), "sip": set()}
    for spec in registry.all_specs():
        by_connectivity[spec.connectivity].add(spec.name)

    # The providers whose calls run over infrastructure the customer owns.
    assert by_connectivity["sip"] == {"cloudonix", "ari"}
    # Everything else is a carrier you buy numbers from, and there is at least
    # one — otherwise the "use a telephony provider" route has nothing to offer.
    assert by_connectivity["api"]


def test_carrier_providers_block_on_a_missing_caller_id():
    """They already refuse to dial without one; readiness must say so too.

    Every carrier's ``validate_config`` requires ``from_numbers``, so before
    this fallback existed a credentials-only configuration reported itself
    ready and then failed the call with a flat "telephony_not_configured".
    """
    covered = []
    for spec in registry.all_specs():
        # Providers with their own resolver are covered by their own tests;
        # this is about everyone falling back to the shared checklist.
        if not spec.requires_caller_id or spec.setup_checklist_resolver:
            continue
        covered.append(spec.name)

        without = get_setup_checklist(
            spec.name,
            {},
            active_phone_number_count=0,
            inbound_routed_phone_number_count=0,
        )
        assert without is not None, spec.name
        assert without.ready_for_outbound is False, spec.name
        assert _step(without, "caller_id").blocks_outbound, spec.name
        # The provider is named from its own metadata, never hardcoded here.
        assert spec.ui_metadata is not None
        assert spec.ui_metadata.display_name in _step(without, "caller_id").description

        with_number = get_setup_checklist(
            spec.name,
            {},
            active_phone_number_count=1,
            inbound_routed_phone_number_count=0,
        )
        assert with_number is not None and with_number.ready_for_outbound, spec.name

    # A guard against the loop silently covering nothing.
    assert covered


def test_provider_that_can_dial_without_a_number_has_no_checklist():
    """ARI originates to an extension; a caller ID is genuinely optional."""
    assert (
        get_setup_checklist(
            "ari",
            {},
            active_phone_number_count=0,
            inbound_routed_phone_number_count=0,
        )
        is None
    )


# RUL-r1-23: a carrier whose call callbacks are verified with a secret its configuration may omit. Without the
# secret every such callback answers 401, so the configuration is not ready for outbound until it has one.
WEBHOOK_SECRETS = {"vonage": "signature_secret", "telnyx": "webhook_public_key"}


@pytest.mark.parametrize(("provider", "field"), sorted(WEBHOOK_SECRETS.items()))
def test_a_configuration_without_its_webhook_secret_is_not_ready_for_outbound(
    provider, field
):
    without = get_setup_checklist(
        provider,
        {field: ""},
        active_phone_number_count=1,
        inbound_routed_phone_number_count=0,
    )
    assert without is not None
    assert without.ready_for_outbound is False
    assert field in without.outbound_blocked_reason

    with_secret = get_setup_checklist(
        provider,
        {field: "configured"},
        active_phone_number_count=1,
        inbound_routed_phone_number_count=0,
    )
    assert with_secret is not None and with_secret.ready_for_outbound


@pytest.mark.parametrize(("provider", "field"), sorted(WEBHOOK_SECRETS.items()))
def test_a_configuration_with_its_webhook_secret_still_needs_a_caller_id(
    provider, field
):
    checklist = get_setup_checklist(
        provider,
        {field: "configured"},
        active_phone_number_count=0,
        inbound_routed_phone_number_count=0,
    )
    assert checklist is not None
    assert checklist.ready_for_outbound is False
    assert _step(checklist, "caller_id").blocks_outbound


def _resolver_db(rows) -> AsyncMock:
    db = AsyncMock()
    db.list_outbound_telephony_configuration_candidates = AsyncMock(return_value=rows)
    db.get_telephony_configuration_for_org = AsyncMock(return_value=rows[0])
    db.list_phone_numbers_for_config = AsyncMock(
        return_value=[
            SimpleNamespace(
                is_active=True, inbound_workflow_id=None, telephony_trunk_id=None
            )
        ]
    )
    db.list_trunks_for_config = AsyncMock(return_value=[])
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "field"), sorted(WEBHOOK_SECRETS.items()))
async def test_dialing_an_explicit_configuration_without_its_webhook_secret_is_setup_incomplete(
    provider, field
):
    row = SimpleNamespace(id=12, name="Carrier", provider=provider, credentials={})

    with pytest.raises(OutboundSetupIncompleteError) as refused:
        await resolve_outbound_configuration_id(12, 7, db=_resolver_db([row]))

    assert field in refused.value.blocked_reason


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "field"), sorted(WEBHOOK_SECRETS.items()))
async def test_the_default_fallback_skips_a_configuration_without_its_webhook_secret(
    provider, field
):
    unverifiable = SimpleNamespace(
        id=12, name="Carrier", provider=provider, credentials={}
    )
    ready = SimpleNamespace(id=13, name="Twilio", provider="twilio", credentials={})

    resolved = await resolve_outbound_configuration_id(
        None, 7, db=_resolver_db([unverifiable, ready])
    )

    assert resolved == ready.id
