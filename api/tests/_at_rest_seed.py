"""Rows with a canary on every registered secret surface, for the encryption-at-rest tests (VOZ-N0-24c).

``DOCUMENTS`` is derived from ``SURFACES``: a surface added to the registry gets a seed without touching the tests.
"""

import re
from uuid import uuid4

from api.db import models as m
from api.enums import WorkflowRunMode
from api.services.security import credential_box
from api.services.security.secret_surfaces import SURFACES

CANARY = "canary-" + "at-rest-" + uuid4().hex[:8]
TOKEN = re.compile(credential_box.TOKEN_PATTERN)

# A concrete step for a registry wildcard, and the keys that hold lists in real graphs (the walk passes through them).
WILDCARDS = {"*": "k", "**": "deep"}
LISTS = {"nodes", "custom_headers"}


async def new_context(session) -> dict:
    """An organization, a user, a workflow and a run to hang rows on."""
    org = m.OrganizationModel(provider_id=f"org-at-rest-{uuid4()}")
    session.add(org)
    await session.flush()
    user = m.UserModel(
        provider_id=f"user-at-rest-{uuid4()}", selected_organization_id=org.id
    )
    session.add(user)
    await session.flush()
    workflow = m.WorkflowModel(name="parent", user_id=user.id, organization_id=org.id)
    session.add(workflow)
    await session.flush()
    run = m.WorkflowRunModel(
        name="parent", workflow_id=workflow.id, mode=WorkflowRunMode.SMALLWEBRTC.value
    )
    session.add(run)
    await session.flush()
    return {"org": org, "user": user, "workflow": workflow, "run": run}


def build_row(ctx, table, column, document, row_key):
    org, user = ctx["org"].id, ctx["user"].id
    build = {
        "organization_configurations": lambda: m.OrganizationConfigurationModel(
            organization_id=org, key=row_key, value=document
        ),
        "telephony_configurations": lambda: m.TelephonyConfigurationModel(
            organization_id=org, name="t", provider="twilio", credentials=document
        ),
        "external_credentials": lambda: m.ExternalCredentialModel(
            organization_id=org, name="c", created_by=user, credential_data=document
        ),
        "user_configurations": lambda: m.UserConfigurationModel(
            user_id=user, key=row_key, configuration=document
        ),
        "integrations": lambda: m.IntegrationModel(
            integration_id="i",
            organization_id=org,
            provider="p",
            connection_details=document,
        ),
        "webhook_deliveries": lambda: m.WebhookDeliveryModel(
            workflow_run_id=ctx["run"].id,
            organization_id=org,
            endpoint_url="https://hooks.example/x",
            webhook_node_id="n",
            custom_headers=document,
        ),
        "workflow_definitions": lambda: m.WorkflowDefinitionModel(
            workflow_id=ctx["workflow"].id, **{column: document}
        ),
        "workflows": lambda: m.WorkflowModel(
            name="w", organization_id=org, **{column: document}
        ),
        "workflow_runs": lambda: m.WorkflowRunModel(
            name="r",
            workflow_id=ctx["workflow"].id,
            mode=WorkflowRunMode.SMALLWEBRTC.value,
            effective_configurations=document,
        ),
        "workflow_templates": lambda: m.WorkflowTemplates(
            template_name="t", template_description="d", template_json=document
        ),
        "tools": lambda: m.ToolModel(
            organization_id=org, name="t", created_by=user, definition=document
        ),
    }
    return build[table]()


def _put(document, path, value):
    node = document
    for part in path[:-1]:
        node = (
            node.setdefault(part, [{}])[0]
            if part in LISTS
            else node.setdefault(part, {})
        )
    node[path[-1]] = value


def documents_for(surfaces) -> dict:
    """One document per (table, column, row key) with a canary at a concrete path of every surface given for it
    (``(table, column, json_path, row_key)``): a string leaf, a dict leaf (``credentials``, holding an ``api_key`` that a ``**`` surface also names), and for
    a whole column a document with nested values."""
    documents: dict = {}
    for index, (table, column, json_path, row_key) in enumerate(surfaces):
        group = (table, column, row_key)
        if not json_path:
            documents[group] = {
                "secret": f"{CANARY}-{index}",
                "nested": {"list": [f"{CANARY}-{index}b"]},
            }
            continue
        path = [WILDCARDS.get(part, part) for part in json_path]
        leaf = (
            {"private_key": f"{CANARY}-{index}", "api_key": f"{CANARY}-{index}n"}
            if path[-1] == "credentials"
            else f"{CANARY}-{index}"
        )
        _put(documents.setdefault(group, {}), path, leaf)
    return documents


DOCUMENTS = documents_for((s.table, s.column, s.json_path, s.row_key) for s in SURFACES)
