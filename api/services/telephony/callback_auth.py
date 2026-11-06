"""Authentication of carrier callbacks: resolve the owning organization's provider and verify the signature, failing
closed with one 401 (VOZ-AC-B4-46b).

A callback is trusted only after the provider of the organization that owns it verified its signature. Every
check here is a read, and every failure is the same 401 (VOZ-AC-B0-28 body), so a forged request moves nothing
and cannot tell a missing transfer from a bad signature. The reason a request was refused goes to the log, never
to the caller, and the log never carries the request's headers or body.
"""

from fastapi import HTTPException
from loguru import logger

from api.db import db_client
from api.services.runtime.cell_state import is_cell
from api.services.telephony.call_transfer_manager import get_call_transfer_manager
from api.services.telephony.carrier_policy import UNVERIFIED_CARRIERS
from api.services.telephony.factory import (
    get_telephony_provider_by_id,
    get_telephony_provider_for_run,
)
from api.services.telephony.transfer_event_protocol import TransferContext
from api.utils.telephony_helper import normalize_webhook_data

UNAUTHENTICATED = "carrier_callback_unauthenticated"


def rejected(
    where: str,
    *,
    why: str,
    code: str = UNAUTHENTICATED,
    reason: str = "the callback could not be authenticated against the telephony credentials of its organization",
    hint: str = "check that the carrier calls the URL this platform configured, and that the auth token or public key in the telephony configuration is the current one",
) -> HTTPException:
    """The 401 for a callback that is not authenticated. ``why`` is for the log only.

    ``where`` and ``why`` are logged as reprs: a request value inside them (a path segment) cannot start a line.
    """
    logger.warning(f"carrier callback refused: code={code} where={where!r} why={why!r}")
    return HTTPException(
        status_code=401,
        detail={"code": code, "where": where, "reason": reason, "hint": hint},
    )


async def provider_for_run(
    workflow_run_id: int,
    *,
    provider_name: str,
    where: str,
    expect_workflow_id: int | None = None,
    expect_organization_id: int | None = None,
):
    """``(run, workflow, provider)``: the provider the run's own organization configured, or a 401.

    The organization comes from the run's workflow, never from the request. ``expect_workflow_id`` and
    ``expect_organization_id`` are ids a request carries outside its signature (an unsigned query): each one
    given must agree with the run, which decides.
    """
    run = await db_client.get_workflow_run_by_id(workflow_run_id)
    workflow = await db_client.get_workflow_by_id(run.workflow_id) if run else None
    if not workflow:
        raise rejected(where, why=f"run {workflow_run_id} or its workflow not found")
    try:
        provider = await get_telephony_provider_for_run(run, workflow.organization_id)
    except Exception as exc:  # any failure to resolve the credentials fails closed
        raise rejected(
            where, why=f"provider unavailable ({type(exc).__name__})"
        ) from exc
    if provider.PROVIDER_NAME != provider_name:
        raise rejected(
            where, why=f"run is on {provider.PROVIDER_NAME}, not {provider_name}"
        )
    if (expect_workflow_id is not None and run.workflow_id != expect_workflow_id) or (
        expect_organization_id is not None
        and workflow.organization_id != expect_organization_id
    ):
        raise rejected(where, why=f"query does not match run {workflow_run_id}")
    return run, workflow, provider


async def provider_for_inbound_webhook(
    provider_class, webhook_data: dict, headers: dict, *, where: str, find_route
):
    """``(normalized_data, provider)``: the provider of the telephony configuration an inbound webhook belongs to,
    or a 401.

    The configuration is the inbound route ``find_route(provider_class, normalized_data)`` matches, never one the
    request names. Any failure to normalize the webhook, match a route or build the provider fails closed.
    """
    try:
        normalized_data = normalize_webhook_data(provider_class, webhook_data, headers)
        match = await find_route(provider_class, normalized_data)
        if not match:
            raise rejected(where, why="no inbound route matches the webhook")
        config, _ = match
        provider = await get_telephony_provider_by_id(config.id, config.organization_id)
    except HTTPException:
        raise
    except Exception as exc:  # any failure to resolve the credentials fails closed
        raise rejected(
            where, why=f"credentials unavailable ({type(exc).__name__})"
        ) from exc
    return normalized_data, provider


async def verify_signature(
    provider, *, where: str, url: str, data: dict, headers: dict, body: str = ""
) -> None:
    """Raise the 401 unless ``provider`` verifies the request. A verifier that raises counts as a failure."""
    try:
        valid = await provider.verify_inbound_signature(url, data, headers, body)
    except Exception as exc:
        raise rejected(
            where, why=f"signature check raised ({type(exc).__name__})"
        ) from exc
    if not valid:
        raise rejected(where, why="signature did not verify")


async def verify_transfer_callback(
    transfer_id: str,
    *,
    provider_name: str,
    where: str,
    url: str,
    data: dict,
    headers: dict,
    body: str = "",
) -> TransferContext:
    """Authenticate a transfer-result callback and return the transfer it is about.

    The credentials come from the TransferContext in Redis: transfer_id -> workflow_run_id -> organization ->
    telephony configuration. A transfer with no context, or a context without a run, has no credentials to
    verify against and is refused.
    """
    context = await (await get_call_transfer_manager()).get_transfer_context(
        transfer_id
    )
    if not context:
        raise rejected(where, why=f"no transfer context for {transfer_id}")
    if not context.workflow_run_id:
        raise rejected(where, why=f"transfer {transfer_id} has no workflow_run_id")
    _, _, provider = await provider_for_run(
        context.workflow_run_id, provider_name=provider_name, where=where
    )
    await verify_signature(
        provider, where=where, url=url, data=data, headers=headers, body=body
    )
    return context


def refuse_unverifiable_signing(provider_name: str, where: str) -> None:
    """Refuse callbacks of a provider whose signature this platform cannot verify.

    A carrier in ``UNVERIFIED_CARRIERS`` is refused in every deployment; any other provider only in a cell, which
    fails closed on a control it cannot enforce (VOZ-AC-B0-30). ``<provider>_signing_unverified`` is the code
    VOZ-N0-40 lists as GATED until the provider's scheme is checked against a real account.
    """
    if provider_name in UNVERIFIED_CARRIERS:
        reason = f"{provider_name} callbacks cannot be authenticated yet, so this platform refuses them"
        hint = f"{provider_name} is disabled until its signature scheme is verified against a real account (VOZ-N0-40)"
    elif is_cell():
        reason = f"{provider_name} callbacks cannot be authenticated yet, and a cell fails closed on a control it cannot enforce"
        hint = f"do not point {provider_name} at a cell until its signature scheme is verified (VOZ-N0-40)"
    else:
        return
    raise rejected(
        where,
        why=f"{provider_name} callbacks carry no signature this platform verifies",
        code=f"{provider_name}_signing_unverified",
        reason=reason,
        hint=hint,
    )
