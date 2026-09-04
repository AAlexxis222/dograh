from dataclasses import dataclass
from typing import Any

from api.services.configuration.cascade import (
    load_effective_workflow_configurations,
)


@dataclass(frozen=True)
class WorkflowRunInputs:
    definition_id: int | None
    initial_context: dict[str, Any]
    effective_configurations: dict[str, Any]
    configuration_warnings: list[str]


def _published_definition(workflow) -> object | None:
    return getattr(workflow, "released_definition", None) or getattr(
        workflow, "current_definition", None
    )


async def prepare_workflow_run_inputs(
    workflow_client,
    workflow,
    *,
    initial_context: dict[str, Any] | None = None,
    use_draft: bool = False,
    include_template_context: bool = False,
) -> WorkflowRunInputs:
    """Resolve definition binding and optional template defaults for a run.

    Draft and template-context handling belong at runtime call sites, not in the
    persistence client. Callers must opt in explicitly for workflow-editor/test
    flows.

    Resolves and returns the effective configuration so every caller freezes
    the same document on the run row (spec §2.2-bis).
    """
    target_definition = None
    if use_draft:
        target_definition = await workflow_client.get_draft_version(workflow.id)

    if target_definition is None:
        target_definition = _published_definition(workflow)

    default_context = {}
    if include_template_context:
        definition_context = (
            getattr(target_definition, "template_context_variables", None)
            if target_definition
            else None
        )
        default_context = (
            definition_context
            if definition_context is not None
            else getattr(workflow, "template_context_variables", None)
        ) or {}

    resolved = await load_effective_workflow_configurations(
        workflow_client,
        organization_id=workflow.organization_id,
        definition_id=getattr(target_definition, "id", None),
    )
    return WorkflowRunInputs(
        definition_id=getattr(target_definition, "id", None),
        initial_context={
            **default_context,
            **(initial_context or {}),
        },
        effective_configurations=resolved.effective,
        configuration_warnings=resolved.warnings,
    )
