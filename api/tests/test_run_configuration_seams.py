"""Two seams read a run's frozen configuration (VOZ-N0-24b fix r1, MAJOR-1 / MINOR-4).

``run_config_without_credentials`` / ``get_workflow_run_config_without_credentials`` hand out the document as stored: credential leaves stay
references or redactions, so a reader that needs no credential (the text-chat inactivity sweep, the external PBX
field mappings before a transfer or a do-not-call) cannot be broken by a reference that no longer resolves.
``run_config_with_credentials`` resolves every credential and fails closed, ``<redacted>`` included.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from api.db.models import OrganizationModel, UserModel
from api.enums import CallType, WorkflowRunMode
from api.services.configuration import cascade
from api.services.configuration.cascade import run_config_with_credentials
from api.services.configuration.credential_refs import (
    REDACTED,
    CredentialRedacted,
    CredentialRefUnresolved,
    credential_ref,
)
from api.services.workflow.run_creation import prepare_workflow_run_inputs
from api.tasks import text_chat_inactivity

CANARY = "canary-seams-0005"
MAPPINGS = [{"source": "rsvp", "field": "comments"}]
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


def _run(effective, *, definition_id=5, definition=None, run_id=1):
    return SimpleNamespace(
        id=run_id,
        effective_configurations=effective,
        definition_id=definition_id,
        definition=definition,
    )


# --- pure seams ---------------------------------------------------------------------------------------------------


def test_settings_keep_references_and_never_need_the_definition():
    frozen = {
        "text_chat_inactivity_timeout_seconds": 120,
        "voicemail_detection": {"api_key": credential_ref(5)},
    }
    settings = cascade.run_config_without_credentials(_run(frozen, definition=None))
    assert settings == frozen and settings is not frozen
    with pytest.raises(CredentialRefUnresolved):
        run_config_with_credentials(_run(frozen, definition=None))


def test_a_redacted_credential_fails_closed_for_a_credential_reader():
    frozen = {"voicemail_detection": {"api_key": REDACTED}}
    with pytest.raises(CredentialRedacted) as raised:
        run_config_with_credentials(_run(frozen))
    error = raised.value
    assert error.code == "credential_redacted"
    assert (
        error.where
        == "workflow_runs:1:effective_configurations:voicemail_detection.api_key"
    )
    assert error.reason and error.hint
    assert cascade.run_config_without_credentials(_run(frozen)) == frozen


def test_settings_of_a_run_without_a_snapshot_are_its_pinned_definition():
    definition = SimpleNamespace(workflow_configurations={"max_call_duration": 60})
    assert cascade.run_config_without_credentials(
        _run(None, definition=definition)
    ) == {"max_call_duration": 60}


# --- the inactivity sweep -------------------------------------------------------------------------------------------


def _session(run_id, effective):
    return SimpleNamespace(
        workflow_run_id=run_id,
        revision=1,
        updated_at=None,
        created_at=None,
        workflow_run=_run(effective, run_id=run_id, definition=None),
    )


async def _sweep(monkeypatch, sessions):
    monkeypatch.setattr(
        text_chat_inactivity.db_client,
        "get_inactive_workflow_run_text_sessions",
        AsyncMock(side_effect=[sessions, []]),
    )
    monkeypatch.setattr(
        text_chat_inactivity,
        "_text_chat_inactivity_deadline",
        lambda session, inactivity_timeout_seconds: text_chat_inactivity.datetime(
            2000, 1, 1, tzinfo=text_chat_inactivity.UTC
        ),
    )
    enqueue = AsyncMock()
    monkeypatch.setattr(
        text_chat_inactivity, "_enqueue_inactive_text_chat_completion", enqueue
    )
    await text_chat_inactivity.sweep_inactive_text_chat_sessions({})
    return [call.kwargs["run_id"] for call in enqueue.await_args_list]


async def test_the_sweep_completes_a_chat_whose_reference_no_longer_resolves(
    monkeypatch,
):
    broken = {"voicemail_detection": {"api_key": credential_ref(5)}}  # definition gone
    assert await _sweep(monkeypatch, [_session(1, broken), _session(2, {})]) == [1, 2]


async def test_one_failing_session_does_not_stop_the_sweep(monkeypatch):
    real = text_chat_inactivity._text_chat_inactivity_timeout_seconds

    def timeout(session):
        if session.workflow_run_id == 1:
            raise RuntimeError("bad session")
        return real(session)

    monkeypatch.setattr(
        text_chat_inactivity, "_text_chat_inactivity_timeout_seconds", timeout
    )
    assert await _sweep(monkeypatch, [_session(1, {}), _session(2, {})]) == [2]


# --- database: a definition that lost the credential ----------------------------------------------------------------


@pytest.fixture
async def broken_run(db_session, async_session):
    org = OrganizationModel(provider_id="test-org-seams")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(provider_id="test-user-seams", selected_organization_id=org.id)
    async_session.add(user)
    await async_session.flush()
    workflow = await db_session.create_workflow(
        name="W", workflow_definition=GRAPH, user_id=user.id, organization_id=org.id
    )
    await db_session.save_workflow_draft(
        workflow.id,
        workflow_configurations={
            "voicemail_detection": {"api_key": f"{CANARY}-vm"},
            "external_pbx_field_mappings": MAPPINGS,
        },
    )
    await db_session.publish_workflow_draft(workflow.id)
    workflow = await db_session.get_workflow(workflow.id, organization_id=org.id)
    run_inputs = await prepare_workflow_run_inputs(db_session, workflow)
    identity = {"type": "vicidial", "callerid": "M1", "lead_id": "42"}
    run = await db_session.create_workflow_run(
        name="run",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        call_type=CallType.OUTBOUND,
        organization_id=org.id,
        definition_id=run_inputs.definition_id,
        initial_context={"external_pbx_call": identity},
        gathered_context={"call_disposition": "DNC", "mapped_call_disposition": "DNC"},
        effective_configurations=run_inputs.effective_configurations,
    )
    # The pinned definition loses the key (e.g. an org model-configuration save rewrote it).
    await async_session.execute(
        text(
            "UPDATE workflow_definitions SET workflow_configurations = CAST(:doc AS json) "
            "WHERE id = :id"
        ),
        {
            "doc": '{"external_pbx_field_mappings": [{"source": "rsvp", "field": "comments"}]}',
            "id": run_inputs.definition_id,
        },
    )
    return org, run, identity


async def test_settings_readers_survive_a_reference_that_no_longer_resolves(
    db_session, broken_run
):
    org, run, _ = broken_run
    settings = await db_session.get_workflow_run_config_without_credentials(
        run.id, org.id
    )
    assert settings["external_pbx_field_mappings"] == MAPPINGS
    assert settings["voicemail_detection"]["api_key"] == credential_ref(
        run.definition_id
    )
    loaded = await db_session.get_workflow_run(run.id, organization_id=org.id)
    assert (
        cascade.run_config_without_credentials(loaded)["external_pbx_field_mappings"]
        == MAPPINGS
    )
    with pytest.raises(CredentialRefUnresolved):
        run_config_with_credentials(loaded)


async def test_pbx_write_back_still_applies_do_not_call(
    db_session, broken_run, monkeypatch
):
    from api.services.telephony import external_pbx_writeback

    _org, run, identity = broken_run
    adapter = SimpleNamespace(
        type="vicidial",
        disposition_fields=lambda disposition: {"status": disposition},
        update_fields=AsyncMock(return_value=SimpleNamespace(ok=True, message="ok")),
        apply_do_not_call=AsyncMock(
            return_value=SimpleNamespace(ok=True, message="ok")
        ),
    )
    monkeypatch.setattr(
        external_pbx_writeback,
        "_adapter_for_organization",
        AsyncMock(return_value=adapter),
    )

    await external_pbx_writeback.sync_external_pbx_call_record(run.id)

    adapter.apply_do_not_call.assert_awaited_once_with(identity, "DNC")
