"""c4a8d2e61f07 (VOZ-N0-30): expansion only. Two nullable columns on workflow_runs, nothing rewritten.

Runs the real DDL inside the test's transaction (the test database is at head): downgrade drops both columns,
upgrade adds them back, and a run that already holds a token keeps it with NULL deadline and NULL revocation.
"""

import ast
from uuid import uuid4

import pytest
from sqlalchemy import text

from api.db.models import (
    OrganizationModel,
    UserModel,
    WorkflowModel,
    WorkflowRunModel,
)
from api.tests._migration_sql import (
    load_migration,
    migration_path,
    run_downgrade,
    run_upgrade,
)

REVISION = "c4a8d2e61f07"
COLUMNS = ("public_access_expires_at", "public_access_revoked_at")


def test_the_migration_only_adds_nullable_columns_without_a_default():
    upgrade = next(
        node
        for node in ast.parse(migration_path(REVISION).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "upgrade"
    )
    calls = [n for n in ast.walk(upgrade) if isinstance(n, ast.Call)]
    ops = [c.func.attr for c in calls if isinstance(c.func, ast.Attribute)]
    assert ops.count("add_column") == 2
    assert set(ops) <= {"add_column", "Column", "DateTime"}
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "Column":
            keywords = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
            assert keywords == {"nullable": True}


def test_the_migration_chains_after_the_credentials_revision():
    assert load_migration(REVISION).down_revision == "e2b7c4d9a613"


async def _columns(async_session) -> dict[str, str]:
    result = await async_session.execute(
        text(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_name = 'workflow_runs' AND column_name = ANY(:names)"
        ),
        {"names": list(COLUMNS)},
    )
    return dict(result.all())


@pytest.mark.asyncio
async def test_downgrade_then_upgrade_adds_the_two_columns_and_keeps_existing_tokens(
    async_session,
):
    migration = load_migration(REVISION)
    organization = OrganizationModel(provider_id=f"links-mig-{uuid4()}")
    async_session.add(organization)
    await async_session.flush()
    user = UserModel(
        provider_id=f"links-mig-{uuid4()}", selected_organization_id=organization.id
    )
    async_session.add(user)
    await async_session.flush()
    workflow = WorkflowModel(
        name="links-mig", user_id=user.id, organization_id=organization.id
    )
    async_session.add(workflow)
    await async_session.flush()
    run = WorkflowRunModel(
        name="legacy",
        workflow_id=workflow.id,
        mode="twilio",
        public_access_token=str(uuid4()),
    )
    async_session.add(run)
    await async_session.flush()

    await run_downgrade(async_session, migration)
    assert await _columns(async_session) == {}

    await run_upgrade(async_session, migration)
    assert await _columns(async_session) == {name: "YES" for name in COLUMNS}
    row = (
        await async_session.execute(
            text(
                "SELECT public_access_token, public_access_expires_at, public_access_revoked_at "
                "FROM workflow_runs WHERE id = :id"
            ),
            {"id": run.id},
        )
    ).one()
    assert row.public_access_token and row.public_access_expires_at is None
    assert row.public_access_revoked_at is None
