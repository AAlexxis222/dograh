"""Gate for organization settings that carry privacy weight: workflow
configuration defaults and Langfuse credentials (VOZ-AC-B0-29).

The operator names the target organization in the path
(/organizations/{organization_id}/...). The routes without an organization in
the path never write: the previous version of the API served them, ignored an
organization_id query parameter and wrote the caller's own organization, so a
write sent there during a rolling deploy or after a rollback could land on the
wrong organization. On the new routes a previous-version pod answers 404.

Fork-only policy, kept out of the shared auth module (depends.py) so upstream
syncs and the future organization role model (VOZ-FU-N0-31-org-roles) touch it
in one place.
"""

from dataclasses import dataclass
from typing import Annotated, NoReturn

from fastapi import Depends, HTTPException, Path, Request

from api.db import db_client
from api.db.models import UserModel
from api.errors.knob_refusal import KnobRefusal
from api.services.auth.depends import get_user

# organizations.id is a 32-bit Integer column: a larger id names nothing, and
# Postgres would refuse it as a parameter instead of returning no row.
_MAX_ORGANIZATION_ID = 2**31 - 1

_TARGET_ORGANIZATION_PATH = Path(
    description="Id of the organization the XPAND operator acts on."
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


def _require_operator(request: Request, user: UserModel, hint: str) -> None:
    """There is no organization role model yet, so the operator is an XPAND
    superuser; real organization roles arrive with the control plane
    (VOZ-FU-N0-31-org-roles)."""
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


async def _resolve_target_organization(
    request: Request, user: UserModel, organization_id: str, hint: str
) -> int:
    """The operator names the organization explicitly: their own selected
    organization is never used, because under local auth nothing switches it
    to a client's."""
    _require_operator(request, user, hint)
    # Parsed here, not by FastAPI, so a client's malformed value stays a 403.
    if not (organization_id.isascii() and organization_id.isdigit()):
        raise _privacy_refusal(
            422,
            "privacy_target_organization_required",
            request,
            reason="an XPAND operator acts on an explicit organization, and "
            "the organization id in the path is not an integer",
            hint="put the target organization's numeric id in the path, "
            "/organizations/{organization_id}/...",
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
            reason="the organization id in the path names no organization",
            hint="check the id of the organization you mean to act on",
        )
    return target


async def get_privacy_operator(
    request: Request,
    organization_id: Annotated[str, _TARGET_ORGANIZATION_PATH],
    user: Annotated[UserModel, Depends(get_user)],
) -> PrivacyTarget:
    """Writes of the privacy settings: an XPAND operator on a named organization."""
    target = await _resolve_target_organization(
        request, user, organization_id, hint="ask an XPAND operator to make this change"
    )
    return PrivacyTarget(operator_id=user.id, organization_id=target)


async def get_privacy_read_organization(
    request: Request,
    organization_id: Annotated[str, _TARGET_ORGANIZATION_PATH],
    user: Annotated[UserModel, Depends(get_user)],
) -> int:
    """Reads of the privacy settings of a named organization: XPAND operators
    only. Everyone reads their own organization on the routes without an
    organization in the path."""
    return await _resolve_target_organization(
        request,
        user,
        organization_id,
        hint="read your own organization's settings on the route without an "
        "organization in the path, or ask an XPAND operator",
    )


async def refuse_untargeted_privacy_write(
    request: Request,
    user: Annotated[UserModel, Depends(get_user)],
) -> NoReturn:
    """The writes without an organization in the path never write: a
    previous-version pod takes the same request as a write on the operator's
    own organization, so here it must not mean anything either."""
    _require_operator(request, user, hint="ask an XPAND operator to make this change")
    collection, setting = request.url.path.rsplit("/", 1)
    raise _privacy_refusal(
        422,
        "privacy_target_organization_required",
        request,
        reason="an XPAND operator names the target organization in the path, "
        "and this route takes none, so it never writes",
        hint=f"use {request.method} {collection}/{{organization_id}}/{setting}",
    )
