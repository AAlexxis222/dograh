import json

from api.tests._migration_sql import load_migration, run_upgrade

strip_secret_leaves = load_migration("d7e3a915c2b8").strip_secret_leaves

SECRET = "sk-test-canary-123"


def test_strip_removes_api_keys_from_both_override_shapes():
    doc = {
        "model_configuration_v2_override": {
            "llm": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": SECRET}
        },
        "model_overrides": {"tts": {"api_key": SECRET, "voice": "x"}},
    }
    out, removed = strip_secret_leaves(doc)
    assert SECRET not in str(out) and removed == 2
    assert out["model_overrides"]["tts"]["voice"] == "x"


def test_resolve_module_no_longer_copies_org_keys():
    import api.services.configuration.resolve as resolve

    assert not hasattr(resolve, "enrich_overrides_with_api_keys")


async def test_migration_strips_secrets_from_frozen_run_snapshots(
    db_session, async_session
):
    """d7e3a915c2b8 on real SQL: runs frozen since a7c2e9d41b06 carry the copied key in
    workflow_runs.effective_configurations, not only in workflows / workflow_definitions."""
    from sqlalchemy import text

    from api.db.models import OrganizationModel, UserModel
    from api.enums import CallType, WorkflowRunMode

    org = OrganizationModel(provider_id="test-org-strip-runs")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(
        provider_id="test-user-strip-runs", selected_organization_id=org.id
    )
    async_session.add(user)
    await async_session.flush()
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition={"nodes": [], "edges": []},
        user_id=user.id,
        organization_id=org.id,
    )
    run = await db_session.create_workflow_run(
        name="run",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        call_type=CallType.INBOUND,
        organization_id=org.id,
    )
    leaked = {
        "max_call_duration": 600,
        "model_configuration_v2_override": {
            "llm": {"provider": "openai", "api_key": SECRET}
        },
    }
    await async_session.execute(
        text(
            "UPDATE workflow_runs SET effective_configurations = CAST(:doc AS json) WHERE id = :id"
        ),
        {"doc": json.dumps(leaked), "id": run.id},
    )

    await run_upgrade(async_session, load_migration("d7e3a915c2b8"))

    stored = (
        await async_session.execute(
            text(
                "SELECT effective_configurations::text FROM workflow_runs WHERE id = :id"
            ),
            {"id": run.id},
        )
    ).scalar_one()
    assert SECRET not in stored
    assert json.loads(stored)["max_call_duration"] == 600
