"""d7e3a915c2b8 on real SQL and realistic documents (review r1 F1).

The migration may only delete a key the runtime can get back from the organization: a legacy
``model_overrides`` leaf whose section runs the organization's own provider with the
organization's own key. ``model_configuration_v2_override`` is self-contained (the runtime
compiles it without the organization) and frozen runs are read again after the call (QA), so
neither is touched.
"""

import copy
import json

import pytest
from sqlalchemy import text

from api.db.models import OrganizationModel, UserModel
from api.enums import CallType, OrganizationConfigurationKey, WorkflowRunMode
from api.schemas.ai_model_configuration import (
    EffectiveAIModelConfiguration,
    OrganizationAIModelConfigurationV2,
)
from api.services.configuration.ai_model_configuration import (
    check_for_masked_keys_in_ai_model_configuration_v2,
    compile_ai_model_configuration_v2,
    convert_legacy_ai_model_configuration_to_v2,
    get_effective_ai_model_configuration_for_workflow,
    mask_ai_model_configuration_v2,
    merge_ai_model_configuration_v2_secrets,
)
from api.tests._migration_sql import load_migration, run_upgrade
from api.tests.integrations._run_pipeline_helpers import USER_CONFIGURATION

ORG_KEY = "sk-org-key-123"
OWN_KEY = "sk-workflow-own-key-456"


def _org_configuration(api_key: str) -> dict:
    """The organization's stored MODEL_CONFIGURATION_V2 (byok pipeline), as the app writes it."""
    legacy = copy.deepcopy(USER_CONFIGURATION)
    for section in ("llm", "tts", "stt"):
        legacy[section]["api_key"] = api_key
    return convert_legacy_ai_model_configuration_to_v2(
        EffectiveAIModelConfiguration.model_validate(legacy)
    ).model_dump(mode="json", exclude_none=True)


def _dograh_override(api_key: str) -> dict:
    return {"version": 2, "mode": "dograh", "dograh": {"api_key": api_key}}


@pytest.fixture
async def org(db_session, async_session):
    organization = OrganizationModel(provider_id="test-org-strip-keys")
    async_session.add(organization)
    await async_session.flush()
    user = UserModel(
        provider_id="test-user-strip-keys", selected_organization_id=organization.id
    )
    async_session.add(user)
    await async_session.flush()
    await db_session.upsert_configuration(
        organization.id,
        OrganizationConfigurationKey.MODEL_CONFIGURATION_V2.value,
        _org_configuration(ORG_KEY),
    )
    return organization, user


async def _workflow_with(db_session, async_session, org, configurations: dict) -> int:
    organization, user = org
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition={"nodes": [], "edges": []},
        user_id=user.id,
        organization_id=organization.id,
    )
    for table, where in (
        ("workflows", "id = :id"),
        ("workflow_definitions", "workflow_id = :id"),
    ):
        await async_session.execute(
            text(
                f"UPDATE {table} SET workflow_configurations = CAST(:doc AS json) "
                f"WHERE {where}"
            ),
            {"doc": json.dumps(configurations), "id": workflow.id},
        )
    return workflow.id


async def _stored(async_session, workflow_id: int) -> list[dict]:
    rows = await async_session.execute(
        text(
            "SELECT workflow_configurations::text FROM workflows WHERE id = :id "
            "UNION ALL SELECT workflow_configurations::text FROM workflow_definitions "
            "WHERE workflow_id = :id"
        ),
        {"id": workflow_id},
    )
    docs = [json.loads(raw) for (raw,) in rows]
    assert len(docs) == 2
    return docs


async def _resolve(org, doc):
    return await get_effective_ai_model_configuration_for_workflow(
        organization_id=org[0].id, workflow_configurations=doc
    )


@pytest.mark.parametrize(
    "override",
    [
        _dograh_override(ORG_KEY),
        _org_configuration(ORG_KEY),  # byok pipeline copied from the org by the PUT
        _org_configuration(OWN_KEY),  # byok pipeline with the workflow's own keys
    ],
    ids=["dograh", "byok_org_copy", "byok_own_keys"],
)
async def test_v2_override_is_untouched_and_still_runs_and_saves(
    db_session, async_session, org, override
):
    configurations = {"model_configuration_v2_override": override}
    workflow_id = await _workflow_with(db_session, async_session, org, configurations)

    await run_upgrade(async_session, load_migration("d7e3a915c2b8"))

    for doc in await _stored(async_session, workflow_id):
        assert doc == configurations
        # The call path compiles the override on its own.
        effective = await _resolve(org, doc)
        assert effective.llm.api_key in (ORG_KEY, OWN_KEY)
        # The PUT path: the client sends the masked document back, secrets are restored
        # from what is stored, and the result must hold no mask.
        existing = OrganizationAIModelConfigurationV2.model_validate(
            doc["model_configuration_v2_override"]
        )
        incoming = OrganizationAIModelConfigurationV2.model_validate(
            mask_ai_model_configuration_v2(existing)
        )
        merged = merge_ai_model_configuration_v2_secrets(incoming, existing)
        check_for_masked_keys_in_ai_model_configuration_v2(merged)
        compile_ai_model_configuration_v2(merged)


async def test_legacy_true_copy_of_the_org_key_is_removed(
    db_session, async_session, org
):
    overrides = {
        "llm": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": ORG_KEY}
    }
    workflow_id = await _workflow_with(
        db_session, async_session, org, {"model_overrides": overrides}
    )

    await run_upgrade(async_session, load_migration("d7e3a915c2b8"))

    for doc in await _stored(async_session, workflow_id):
        assert "api_key" not in doc["model_overrides"]["llm"]
        assert doc["model_overrides"]["llm"]["model"] == "gpt-4.1-mini"
        effective = await _resolve(org, doc)
        # Same provider: the runtime merges onto the org's section, key included.
        assert (effective.llm.model, effective.llm.api_key) == ("gpt-4.1-mini", ORG_KEY)


@pytest.mark.parametrize(
    "overrides",
    [
        # Another provider: the runtime builds the section from the override alone.
        {"tts": {"provider": "elevenlabs", "api_key": ORG_KEY, "voice": "v1"}},
        # The org's provider, but the workflow's own key: not a copy.
        {"llm": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": OWN_KEY}},
    ],
    ids=["cross_provider", "own_key"],
)
async def test_legacy_key_that_is_not_a_copy_is_kept(
    db_session, async_session, org, overrides
):
    configurations = {"model_overrides": overrides}
    workflow_id = await _workflow_with(db_session, async_session, org, configurations)

    await run_upgrade(async_session, load_migration("d7e3a915c2b8"))

    for doc in await _stored(async_session, workflow_id):
        assert doc == configurations
        await _resolve(org, doc)


async def test_frozen_runs_are_untouched(db_session, async_session, org):
    organization, user = org
    workflow_id = await _workflow_with(db_session, async_session, org, {})
    run = await db_session.create_workflow_run(
        name="run",
        workflow_id=workflow_id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        call_type=CallType.INBOUND,
        organization_id=organization.id,
    )
    frozen = {
        "model_overrides": {"llm": {"provider": "openai", "api_key": ORG_KEY}},
        "model_configuration_v2_override": _dograh_override(ORG_KEY),
    }
    await async_session.execute(
        text(
            "UPDATE workflow_runs SET effective_configurations = CAST(:doc AS json), "
            "state = 'completed' WHERE id = :id"
        ),
        {"doc": json.dumps(frozen), "id": run.id},
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
    assert json.loads(stored) == frozen


async def test_guard_aborts_when_a_true_copy_is_still_stored(
    db_session, async_session, org
):
    overrides = {
        "llm": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": ORG_KEY}
    }
    await _workflow_with(db_session, async_session, org, {"model_overrides": overrides})
    migration = load_migration("d7e3a915c2b8")
    # A pass whose writes do not land: the guard re-reads the store and finds the copy.
    migration._write = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="still hold"):
        await run_upgrade(async_session, migration)
