from unittest.mock import AsyncMock

import pytest

import scripts.backfill_workflow_configuration_defaults as backfill
from api.db.models import OrganizationModel, UserModel
from scripts.backfill_workflow_configuration_defaults import (
    run_backfill,
    strip_schema_default_leaves,
)

GRAPH = {
    "nodes": [{"id": "1", "type": "startCall", "data": {"name": "S", "prompt": "p"}}],
    "edges": [],
}


def test_strip_uses_pydantic_normalised_equality_and_keeps_nested_null():
    baked = {
        "max_call_duration": 300,
        "dictionary": "",
        "ambient_noise_configuration": {"enabled": False, "volume": 0.3},
        "call_dispositions": [],
        "user_turn_stop_timeout": None,
        "turn": {"analyzer": {"url": None}},
        "smart_turn_stop_secs": 3.5,
    }
    sparse, removed = strip_schema_default_leaves(baked)
    assert sparse == {"turn": {"analyzer": {"url": None}}, "smart_turn_stop_secs": 3.5}
    assert "user_turn_stop_timeout" in removed and "dictionary" in removed
    assert strip_schema_default_leaves(sparse)[0] == sparse  # idempotent


def test_strip_keeps_partial_sections_and_out_of_schema_keys():
    sparse, _ = strip_schema_default_leaves(
        {
            "ambient_noise_configuration": {"enabled": True, "volume": 0.3},
            "knowledge_base": {"top_k": 3},
        }
    )
    assert sparse == {
        "ambient_noise_configuration": {"enabled": True},
        "knowledge_base": {"top_k": 3},
    }


async def test_backfill_reports_and_skips_a_non_object_document(monkeypatch):
    """A definition whose configurations are a list (or anything but an object)
    must not stop the sweep: it is reported and the next row still runs."""
    batches = [
        [
            (11, 1, 5, ["not", "an", "object"]),
            (12, 2, 5, {"max_call_duration": 300, "dictionary": "mine"}),
        ],
        [],
    ]
    monkeypatch.setattr(
        backfill.db_client,
        "list_draft_definitions_for_backfill",
        AsyncMock(side_effect=batches),
    )
    writes = AsyncMock()
    monkeypatch.setattr(backfill.db_client, "update_definition_configurations", writes)

    report = await run_backfill(apply=True)

    assert report[5][0] == {
        "workflow_id": 1,
        "definition_id": 11,
        "skipped": "not an object",
    }
    assert report[5][1]["removed"] == ["max_call_duration"]
    writes.assert_awaited_once_with(12, {"dictionary": "mine"})


@pytest.fixture
async def org_user(async_session):
    org = OrganizationModel(provider_id="test-org-backfill")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(provider_id="test-user-backfill", selected_organization_id=org.id)
    async_session.add(user)
    await async_session.flush()
    return org, user


async def test_backfill_is_idempotent_and_touches_drafts_only(db_session, org_user):
    org, user = org_user
    baked = {"max_call_duration": 300, "dictionary": "mine", "call_dispositions": []}
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition=GRAPH,
        user_id=user.id,
        organization_id=org.id,
    )
    await db_session.save_workflow_draft(workflow.id, workflow_configurations=baked)
    # published V? keeps the baked doc
    await db_session.publish_workflow_draft(workflow.id)
    # new draft
    await db_session.save_workflow_draft(
        workflow.id, workflow_configurations=dict(baked)
    )

    dry = await run_backfill(apply=False)
    draft = await db_session.get_draft_version(workflow.id)
    assert draft.workflow_configurations == baked  # dry-run writes nothing
    assert dry[org.id][0]["removed"] == ["call_dispositions", "max_call_duration"]

    await run_backfill(apply=True)
    draft = await db_session.get_draft_version(workflow.id)
    assert draft.workflow_configurations == {"dictionary": "mine"}
    versions = await db_session.get_workflow_versions(workflow.id)
    published = next(v for v in versions if v.status == "published")
    assert published.workflow_configurations == baked  # snapshots untouched

    again = await run_backfill(apply=True)
    assert again == {} or all(
        row["removed"] == [] for rows in again.values() for row in rows
    )


async def test_update_definition_configurations_writes_drafts_only(
    db_session, org_user
):
    """The write seam refuses a published definition even when handed its id:
    one published between the listing and the UPDATE is a run snapshot by then.
    """
    org, user = org_user
    baked = {"max_call_duration": 300, "dictionary": "mine"}
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition=GRAPH,
        user_id=user.id,
        organization_id=org.id,
    )
    await db_session.save_workflow_draft(workflow.id, workflow_configurations=baked)
    await db_session.publish_workflow_draft(workflow.id)
    await db_session.save_workflow_draft(
        workflow.id, workflow_configurations=dict(baked)
    )
    versions = await db_session.get_workflow_versions(workflow.id)
    published = next(v for v in versions if v.status == "published")
    draft = next(v for v in versions if v.status == "draft")

    await db_session.update_definition_configurations(published.id, {"dictionary": "x"})
    await db_session.update_definition_configurations(draft.id, {"dictionary": "x"})

    versions = {v.id: v for v in await db_session.get_workflow_versions(workflow.id)}
    assert versions[published.id].workflow_configurations == baked
    assert versions[draft.id].workflow_configurations == {"dictionary": "x"}
