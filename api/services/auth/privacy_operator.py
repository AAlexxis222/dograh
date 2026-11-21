"""Gate for organization settings that carry privacy weight: workflow
configuration defaults and Langfuse credentials (VOZ-AC-B0-29).

Fork-only policy, kept out of the shared auth module (depends.py) so upstream
syncs and the future organization role model (VOZ-FU-N0-31-org-roles) touch it
in one place.
"""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Request

from api.db import db_client
from api.db.models import UserModel
from api.errors.knob_refusal import KnobRefusal
from api.services.auth.depends import get_user, get_user_with_selected_organization

# organizations.id is a 32-bit Integer column: a larger id names nothing, and
# Postgres would refuse it as a parameter instead of returning no row.
_MAX_ORGANIZATION_ID = 2**31 - 1

_TARGET_ORGANIZATION_QUERY = Query(
    description="Target organization. XPAND operators only: required on writes; "
    "on reads, omit it to read your own organization."
)


@dataclass(frozen=True)
class PrivacyTarget:
    """The operator acting and the organization they act on."""

    operator_id: int
    organization_id: int


def _privacy_refusal(
    status: int, code: str, request: Request, reason: str, hint: str
) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail=KnobRefusal(
            code=code,
            where=f"{request.method} {request.url.path}",
            reason=reason,
            hint=hint,
        ).as_detail(),
    )


async def _resolve_target_organization(
    request: Request, user: UserModel, organization_id: str | None, hint: str
) -> int:
    """There is no organization role model yet, so the operator is an XPAND
    superuser; real organization roles arrive with the control plane
    (VOZ-FU-N0-31-org-roles). The operator names the organization explicitly:
    their own selected organization is never used, because under local auth
    nothing switches it to a client's.
    """
    if not user.is_superuser:
        raise _privacy_refusal(
            403,
            "privacy_knob_forbidden",
            request,
            reason="only XPAND operators change an organization's workflow "
            "configuration defaults and telemetry credentials, or read them "
            "for a named organization",
            hint=hint,
        )
    # Parsed here, not by FastAPI, so a client's malformed value stays a 403.
    if not (
        organization_id and organization_id.isascii() and organization_id.isdigit()
    ):
        raise _privacy_refusal(
            422,
            "privacy_target_organization_required",
            request,
            reason="an XPAND operator acts on an explicit organization, and "
            "organization_id is missing or not an integer",
            hint="pass the target organization's id as organization_id",
        )
    digits = organization_id.lstrip("0") or "0"
    # Length first: int() refuses strings past its digit limit with an error.
    target = int(digits) if len(digits) <= 10 else _MAX_ORGANIZATION_ID + 1
    if target > _MAX_ORGANIZATION_ID or not await db_client.get_organization_by_id(
        target
    ):
        raise _privacy_refusal(
            404,
            "organization_not_found",
            request,
            reason="organization_id names no organization",
            hint="check the id of the organization you mean to act on",
        )
    return target


async def get_privacy_operator(
    request: Request,
    user: Annotated[UserModel, Depends(get_user)],
    organization_id: Annotated[str | None, _TARGET_ORGANIZATION_QUERY] = None,
) -> PrivacyTarget:
    """Writes of the privacy settings: an XPAND operator on a named organization."""
    target = await _resolve_target_organization(
        request, user, organization_id, hint="ask an XPAND operator to make this change"
    )
    return PrivacyTarget(operator_id=user.id, organization_id=target)


async def get_privacy_read_organization(
    request: Request,
    user: Annotated[UserModel, Depends(get_user)],
    organization_id: Annotated[str | None, _TARGET_ORGANIZATION_QUERY] = None,
) -> int:
    """Reads of the privacy settings: the caller's own organization, as before,
    or, for an XPAND operator only, the organization named by organization_id."""
    if organization_id is None:
        return (
            await get_user_with_selected_organization(user)
        ).selected_organization_id
    return await _resolve_target_organization(
        request,
        user,
        organization_id,
        hint="omit organization_id to read your own organization's settings, "
        "or ask an XPAND operator",
    )
