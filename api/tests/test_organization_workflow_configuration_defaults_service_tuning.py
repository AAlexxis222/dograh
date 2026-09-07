import pytest

from api.services.organization_workflow_configuration_defaults import (
    OrganizationWorkflowConfigurationRejected,
    validate_organization_workflow_configuration_document,
)


def test_organization_defaults_reject_service_tuning_ctor_url():
    with pytest.raises(
        OrganizationWorkflowConfigurationRejected,
        match="service_tuning.stt.deepgram.ctor.url",
    ):
        validate_organization_workflow_configuration_document(
            {"service_tuning": {"stt": {"deepgram": {"ctor": {"url": "wss://x"}}}}}
        )
