"""Public download endpoints for workflow recordings and transcripts.

These endpoints provide secure, token-based public access to workflow artifacts
without requiring authentication. Tokens are generated on-demand during
post-call processing for runs that execute integrations, QA, or campaign
reporting. A link expires ``PUBLIC_LINK_TTL_H`` hours after it was minted, can be revoked, and is off inside a
cell unless the organization turned it on (VOZ-N0-30).
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from loguru import logger

from api.db import db_client
from api.services import public_links
from api.services.pipecat.openai_efforts import KnobRefusal
from api.services.storage import get_storage_for_backend
from api.utils.recording_artifacts import (
    get_recording_storage_backend,
    get_recording_storage_key,
)

router = APIRouter(prefix="/public/download")

_WHERE = "GET /public/download/workflow/{token}/{artifact_type}"


def _link_refused(code: str, reason: str, hint: str) -> HTTPException:
    """The one 404 of a link that does not open, in the VOZ-AC-B0-28 shape (R3: KnobRefusal, no new error class)."""
    detail = KnobRefusal(code=code, where=_WHERE, reason=reason, hint=hint).as_detail()
    return HTTPException(status_code=404, detail=detail)


@router.get("/workflow/{token}/{artifact_type}")
async def download_workflow_artifact(
    token: str,
    artifact_type: str,
    inline: bool = Query(
        default=False, description="Display inline in browser instead of download"
    ),
):
    """Download a workflow recording or transcript via public access token.

    This endpoint:
    1. Validates the public access token
    2. Looks up the corresponding workflow run
    3. Generates a signed URL for the requested artifact
    4. Redirects to the signed URL

    Args:
        token: The public access token (UUID format)
        artifact_type: Type of artifact - "recording", "transcript",
            "user_recording", or "bot_recording"
        inline: If true, sets Content-Disposition to inline for browser preview

    Returns:
        RedirectResponse to the signed URL (302 redirect)

    Raises:
        HTTPException 400: If artifact type is unsupported
        HTTPException 404: If token is invalid, expired or its links are off, or artifact not found
    """
    # 1. Lookup workflow run by token
    workflow_run = await db_client.get_workflow_run_by_public_token(token)
    if not workflow_run:
        logger.warning(f"Invalid public access token: {token[:8]}...")
        raise _link_refused(
            "link_invalid",
            "no public link matches this token (never issued, or revoked)",
            "open the run in the app to see its current links; a revoked link is never issued again",
        )

    if not await db_client.public_links_enabled_for_run(workflow_run.id):
        logger.warning(f"Public links are off: workflow_run_id={workflow_run.id}")
        raise _link_refused(
            "public_links_disabled",
            "public download links are off for this organization",
            "an operator turns them on with public_links_enabled: true in the organization's "
            "workflow-configuration-defaults",
        )

    if public_links.is_expired(
        workflow_run.public_access_expires_at, public_links.now()
    ):
        logger.warning(f"Expired public link: workflow_run_id={workflow_run.id}")
        raise _link_refused(
            "link_expired",
            "this public link has expired",
            "links live PUBLIC_LINK_TTL_H hours from when they were issued; use the authenticated API for "
            "the recording or transcript",
        )

    # 2. Get file path based on artifact type
    artifact_storage_backend = None
    if artifact_type == "recording":
        file_path = workflow_run.recording_url
    elif artifact_type == "transcript":
        file_path = workflow_run.transcript_url
    elif artifact_type == "user_recording":
        file_path = get_recording_storage_key(workflow_run.extra, "user")
        artifact_storage_backend = get_recording_storage_backend(
            workflow_run.extra, "user"
        )
    elif artifact_type == "bot_recording":
        file_path = get_recording_storage_key(workflow_run.extra, "bot")
        artifact_storage_backend = get_recording_storage_backend(
            workflow_run.extra, "bot"
        )
    else:
        logger.warning(
            f"Unsupported artifact type: type={artifact_type}, workflow_run_id={workflow_run.id}"
        )
        raise HTTPException(status_code=400, detail="Unsupported artifact type")

    if not file_path:
        logger.warning(
            f"Artifact not found: type={artifact_type}, workflow_run_id={workflow_run.id}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"No {artifact_type} available for this workflow run",
        )

    # 3. Get storage backend for this workflow run
    try:
        storage = get_storage_for_backend(
            artifact_storage_backend or workflow_run.storage_backend
        )
    except ValueError as e:
        logger.error(f"Invalid storage backend: {workflow_run.storage_backend}")
        raise HTTPException(status_code=500, detail="Storage configuration error")

    # 4. Generate signed URL (1 hour expiration)
    try:
        signed_url = await storage.aget_signed_url(
            file_path=file_path,
            expiration=3600,  # 1 hour
            force_inline=inline,
        )
    except Exception as e:
        logger.error(f"Failed to generate signed URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate download URL")

    if not signed_url:
        logger.error(f"Storage returned None for signed URL: {file_path}")
        raise HTTPException(status_code=500, detail="Failed to generate download URL")

    logger.info(
        f"Generated signed URL for {artifact_type}: workflow_run_id={workflow_run.id}, token={token[:8]}..."
    )

    # 5. Redirect to signed URL
    return RedirectResponse(url=signed_url, status_code=302)
