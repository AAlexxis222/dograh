"""A run stores credential references, never credential values (VOZ-N0-24b; VOZ-AC-B1-07, VOZ-AT-B6-05 (c)).

Every runtime reader of a run's frozen configuration gets the real value back through the resolver, at the two read
points: ``run_configurations_for`` (pipeline, QA, quota, text chat, inactivity) and
``get_workflow_run_configurations`` (PBX write-back, custom tools).
"""

import json

import pytest
from sqlalchemy import text

from api.db.models import OrganizationModel, UserModel
from api.enums import CallType, WorkflowRunMode
from api.services.configuration.ai_model_configuration import (
    get_effective_ai_model_configuration_for_workflow,
)
from api.services.configuration.cascade import run_configurations_for
from api.services.configuration.credential_refs import (
    REDACTED,
    CredentialRefUnresolved,
    credential_ref,
    is_credential_marker,
    resolve_credential_refs,
    stored_secret_paths,
    to_credential_refs,
)
from api.services.security.secret_surfaces import WORKFLOW_CONFIGURATION_SECRET_PATHS
from api.services.security.secret_walk import iter_leaves
from api.services.workflow.run_creation import prepare_workflow_run_inputs

CANARY = "canary-run-credential-0001"

GRAPH = {
    "nodes": [
        {"id": "1", "type": "startCall", "data": {"name": "Start", "prompt": "Hi"}},
        {"id": "2", "type": "endCall", "data": {"name": "End", "prompt": "Bye"}},
    ],
    "edges": [
        {
            "id": "e",
            "source": "1",
            "target": "2",
            "data": {"label": "end", "condition": "x"},
        }
    ],
}


def _secrets_document(tag: str = CANARY) -> dict:
    """A workflow configuration with a value at every registered secret path shape: strings, a key list, a
    credentials object with a key nested in it, keys inside a list, a voicemail key and a service-tuning URL."""
    return {
        "model_configuration_v2_override": {
            "byok": {
                "stt": {"credentials": {"api_key": f"{tag}-v2-nested"}},
                "fallbacks": [{"provider": "a", "api_key": f"{tag}-v2-listed"}],
            }
        },
        "model_overrides": {
            "llm": {"provider": "openai", "api_key": f"{tag}-llm"},
            "tts": {
                "provider": "elevenlabs",
                "api_key": [f"{tag}-tts-1", f"{tag}-tts-2"],
            },
            "stt": {"provider": "google", "credentials": {"api_key": f"{tag}-nested"}},
            "realtime": {"aws_secret_key": f"{tag}-aws", "aws_access_key": ""},
        },
        "voicemail_detection": {"enabled": True, "api_key": f"{tag}-vm"},
        "service_tuning": {"stt": {"deepgram": {"ctor": {"url": f"wss://{tag}-url"}}}},
    }


def _registered_leaves(document: dict) -> list:
    return [
        (path, container[key])
        for pattern in WORKFLOW_CONFIGURATION_SECRET_PATHS
        for path, container, key in iter_leaves(document, pattern, (), indexed=True)
    ]


# --- pure functions -----------------------------------------------------------------------------------------------


def test_every_registered_leaf_becomes_a_reference_and_no_value_survives():
    document = _secrets_document()
    frozen = to_credential_refs(document, document, 42)

    assert CANARY not in json.dumps(frozen)
    leaves = _registered_leaves(frozen)
    assert leaves, "the fixture must exercise the registry"
    for path, value in leaves:
        assert value in ("", None) or value == credential_ref(42), path
    assert stored_secret_paths(frozen) == []
    assert frozen["model_overrides"]["llm"]["provider"] == "openai"  # non-secrets kept
    assert frozen["model_overrides"]["realtime"]["aws_access_key"] == ""  # empties kept


def test_a_value_the_pinned_definition_does_not_hold_is_redacted():
    effective = _secrets_document()
    pinned = _secrets_document("other")
    pinned["voicemail_detection"]["api_key"] = f"{CANARY}-vm"

    frozen = to_credential_refs(effective, pinned, 7)

    assert frozen["voicemail_detection"]["api_key"] == credential_ref(7)
    assert frozen["model_overrides"]["llm"]["api_key"] == REDACTED
    assert CANARY not in json.dumps(frozen)
    assert (
        to_credential_refs(effective, None, 7)["voicemail_detection"]["api_key"]
        == REDACTED
    )
    assert (
        to_credential_refs(effective, effective, None)["voicemail_detection"]["api_key"]
        == REDACTED
    )


def test_resolution_gives_back_the_definition_values():
    document = _secrets_document()
    frozen = to_credential_refs(document, document, 42)

    assert (
        resolve_credential_refs(frozen, lambda: document, definition_id=42, run_id=1)
        == document
    )


def test_resolution_leaves_plain_values_and_redactions_and_skips_the_definition():
    legacy = _secrets_document()
    legacy["voicemail_detection"]["api_key"] = REDACTED

    def definition_not_needed():
        raise AssertionError("a run without references must not load its definition")

    resolved = resolve_credential_refs(
        legacy, definition_not_needed, definition_id=42, run_id=1
    )
    assert resolved == legacy
    assert resolved is not legacy


@pytest.mark.parametrize(
    ("definition_id", "pinned", "reason"),
    [
        (43, _secrets_document(), "names workflow definition 42"),
        (42, {}, "holds no value there"),
        (42, None, "holds no value there"),
        (42, {"voicemail_detection": {"api_key": ""}}, "holds no value there"),
    ],
    ids=["other_definition", "leaf_gone", "definition_gone", "leaf_emptied"],
)
def test_an_unresolvable_reference_fails_closed_in_b0_28_shape(
    definition_id, pinned, reason
):
    frozen = {"voicemail_detection": {"api_key": credential_ref(42)}}

    with pytest.raises(CredentialRefUnresolved) as raised:
        resolve_credential_refs(
            frozen, lambda: pinned, definition_id=definition_id, run_id=9
        )

    error = raised.value
    assert error.code == "credential_ref_unresolved"
    assert (
        error.where
        == "workflow_runs:9:effective_configurations:voicemail_detection.api_key"
    )
    assert reason in error.reason and error.hint
    assert CANARY not in str(error)


def test_markers_are_recognised_and_values_are_not():
    assert is_credential_marker(REDACTED) and is_credential_marker(credential_ref(5))
    assert not is_credential_marker("<credential_ref:workflow_definitions:x>")
    assert not is_credential_marker(f"{credential_ref(5)} ")
    assert not is_credential_marker(CANARY) and not is_credential_marker(["<redacted>"])


# --- database: run creation and every reader ----------------------------------------------------------------------


@pytest.fixture
async def org_user(async_session):
    org = OrganizationModel(provider_id="test-org-credrefs")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(provider_id="test-user-credrefs", selected_organization_id=org.id)
    async_session.add(user)
    await async_session.flush()
    return org, user


async def _run_with(db_session, org, user, configurations):
    workflow = await db_session.create_workflow(
        name="W", workflow_definition=GRAPH, user_id=user.id, organization_id=org.id
    )
    await db_session.save_workflow_draft(
        workflow.id, workflow_configurations=configurations
    )
    await db_session.publish_workflow_draft(workflow.id)
    workflow = await db_session.get_workflow(workflow.id, organization_id=org.id)
    run_inputs = await prepare_workflow_run_inputs(db_session, workflow)
    run = await db_session.create_workflow_run(
        name="run",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        call_type=CallType.INBOUND,
        organization_id=org.id,
        definition_id=run_inputs.definition_id,
        initial_context=run_inputs.initial_context,
        effective_configurations=run_inputs.effective_configurations,
    )
    return run, run_inputs


def _v2_override(api_key: str) -> dict:
    return {"version": 2, "mode": "dograh", "dograh": {"api_key": api_key}}


async def test_a_new_run_stores_references_and_every_reader_resolves_them(
    db_session, async_session, org_user
):
    org, user = org_user
    configurations = {
        "model_configuration_v2_override": _v2_override(f"{CANARY}-v2"),
        "voicemail_detection": {"enabled": True, "api_key": f"{CANARY}-vm"},
    }
    run, run_inputs = await _run_with(db_session, org, user, configurations)

    # At rest: no value in the run row, in any form.
    raw = (
        await async_session.execute(
            text(
                "SELECT effective_configurations::text FROM workflow_runs WHERE id = :id"
            ),
            {"id": run.id},
        )
    ).scalar_one()
    assert CANARY not in raw
    stored = json.loads(raw)
    assert stored_secret_paths(stored) == []
    assert stored["voicemail_detection"]["api_key"] == credential_ref(
        run_inputs.definition_id
    )
    assert CANARY not in json.dumps(run_inputs.effective_configurations)

    # Column-select reader (PBX write-back, custom tools).
    by_columns = await db_session.get_workflow_run_configurations(run.id, org.id)
    assert by_columns["voicemail_detection"]["api_key"] == f"{CANARY}-vm"

    # ORM readers: the pipeline/quota loader and the QA/text-chat loader.
    for loaded in (
        await db_session.get_workflow_run(run.id, organization_id=org.id),
        (await db_session.get_workflow_run_with_context(run.id))[0],
    ):
        resolved = run_configurations_for(loaded)
        assert resolved["voicemail_detection"]["api_key"] == f"{CANARY}-vm"
        # ai_model_configuration.py:85-106 compiles the frozen v2 override with its key.
        effective = await get_effective_ai_model_configuration_for_workflow(
            organization_id=org.id, workflow_configurations=resolved
        )
        assert effective.llm.api_key == f"{CANARY}-v2"


async def test_the_store_refuses_a_credential_value_in_a_run_row(db_session, org_user):
    org, user = org_user
    run, run_inputs = await _run_with(db_session, org, user, {})

    with pytest.raises(RuntimeError, match="voicemail_detection.api_key"):
        await db_session.create_workflow_run(
            name="leak",
            workflow_id=run.workflow_id,
            mode=WorkflowRunMode.SMALLWEBRTC.value,
            user_id=user.id,
            organization_id=org.id,
            definition_id=run_inputs.definition_id,
            effective_configurations={"voicemail_detection": {"api_key": CANARY}},
        )


async def test_a_reference_whose_definition_lost_the_value_fails_closed(
    db_session, async_session, org_user
):
    org, user = org_user
    run, run_inputs = await _run_with(
        db_session, org, user, {"voicemail_detection": {"api_key": f"{CANARY}-vm"}}
    )
    await async_session.execute(
        text(
            "UPDATE workflow_definitions SET workflow_configurations = '{}' WHERE id = :id"
        ),
        {"id": run_inputs.definition_id},
    )

    with pytest.raises(CredentialRefUnresolved):
        await db_session.get_workflow_run_configurations(run.id, org.id)
    with pytest.raises(CredentialRefUnresolved):
        run_configurations_for(
            await db_session.get_workflow_run(run.id, organization_id=org.id)
        )
