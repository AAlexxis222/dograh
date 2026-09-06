"""The frozen run document's service_tuning reaches the factories, and the
extraction LLM stops sharing the conversation instance when scope.extraction
is false. Mold: integrations/test_run_pipeline.py (fixture + boot/teardown)."""

import asyncio
from unittest.mock import patch

import pytest
from pipecat.tests.mock_transport import MockTransport
from pipecat.transports.base_transport import TransportParams

from api.enums import WorkflowRunMode
from api.services.pipecat.audio_config import create_audio_config
from api.services.pipecat.run_pipeline import _run_pipeline
from api.services.pipecat.worker_runner import wait_for_pipeline_worker_started
from api.tests.integrations._run_pipeline_helpers import (
    PassthroughProcessor,
    create_workflow_run_rows,
    patch_run_pipeline_externals,
)
from api.tests.integrations.test_run_pipeline import WORKFLOW_DEFINITION
from pipecat.tests import MockLLMService, MockTTSService

TUNING = {
    "stt": {"deepgram": {"settings": {"numerals": True}}},
    "llm": {"_all": {"settings": {"temperature": 0.4}}},
}
# Terminal classification is what makes _run_pipeline ask for an extraction
# LLM at all; without it the conversation instance is simply reused.
CALL_DISPOSITIONS = [
    {"code": "qualified", "description": "The call achieved its goal."}
]


@pytest.fixture
async def workflow_run_setup(db_session, async_session):
    return await create_workflow_run_rows(
        db_session,
        async_session,
        workflow_definition=WORKFLOW_DEFINITION,
        name_prefix="Service Tuning Integration",
        provider_id_suffix="service-tuning",
        organization_configuration_defaults={
            "service_tuning": TUNING,
            "call_dispositions": CALL_DISPOSITIONS,
        },
    )


async def _boot_and_stop(workflow_run, user, workflow, captured_task):
    transport = MockTransport(
        TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_end_silence_secs=0,
        )
    )
    run_task = asyncio.create_task(
        _run_pipeline(
            transport=transport,
            workflow_id=workflow.id,
            workflow_run_id=workflow_run.id,
            user_id=user.id,
            audio_config=create_audio_config(WorkflowRunMode.SMALLWEBRTC.value),
            user_provider_id=user.provider_id,
        )
    )
    for _ in range(60):
        if captured_task or run_task.done():
            break
        await asyncio.sleep(0.05)
    if run_task.done() and not captured_task:
        run_task.result()
    assert captured_task, "create_pipeline_task was never invoked"
    await wait_for_pipeline_worker_started(
        captured_task[0], timeout=3.0, run_task=run_task
    )
    await captured_task[0].cancel()
    await asyncio.wait_for(run_task, timeout=5.0)


@pytest.mark.asyncio
async def test_factories_receive_service_tuning_from_the_frozen_run(
    workflow_run_setup, db_session
):
    workflow_run, user, workflow = workflow_run_setup
    captured_task: list = []
    stt_calls, tts_calls = [], []
    with (
        patch_run_pipeline_externals(captured_task),
        patch(
            "api.services.pipecat.run_pipeline.create_stt_service",
            side_effect=lambda *a, **k: stt_calls.append(k) or PassthroughProcessor(),
        ),
        patch(
            "api.services.pipecat.run_pipeline.create_tts_service",
            side_effect=lambda *a, **k: tts_calls.append(k) or MockTTSService(),
        ),
    ):
        await _boot_and_stop(workflow_run, user, workflow, captured_task)
    assert stt_calls[0]["tuning"] == TUNING and tts_calls[0]["tuning"] == TUNING


@pytest.mark.asyncio
async def test_extraction_llm_is_requested_with_its_own_role_when_scope_excludes_it(
    workflow_run_setup, db_session
):
    workflow_run, user, workflow = workflow_run_setup
    captured_task: list = []
    llm_calls = []
    with (
        patch_run_pipeline_externals(captured_task),
        patch(
            "api.services.pipecat.run_pipeline.create_llm_service",
            side_effect=lambda *a, **k: (
                llm_calls.append(k) or MockLLMService(api_key="test")
            ),
        ),
    ):
        await _boot_and_stop(workflow_run, user, workflow, captured_task)
    roles = [c.get("role", "conversation") for c in llm_calls]
    # Without llm tuning (or with scope.extraction) there is no "extraction" call.
    assert roles.count("conversation") == 1 and "extraction" in roles
    assert all(c["tuning"] == TUNING for c in llm_calls)
