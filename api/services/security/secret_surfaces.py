"""The one registry of where a credential can live at rest (VOZ-AC-B6-15, inventory of VOZ-AC-B0-16).

A surface is ``(table, column, json_path)``, plus ``row_key`` when a keyed store holds a different document per row
(the ``key`` column of organization_configurations). ``json_path`` is a tuple of keys: ``()`` is the whole column
value, ``*`` matches one level and ``**`` any depth, and either passes through a list (the walk of
api/services/security/secret_walk.py). Encryption at rest and VOZ-AT-B6-05 derive from this list.

``SECRET_PATHS`` (secrets_registry.py) derives from it directly. The org model-configuration masking reads
``SECRET_LEAF_NAMES`` from here, and the node and call-event secret fields that integration packages declare at load
time are guarded against it by api/tests/test_credential_box.py.
"""

from dataclasses import dataclass

from api.enums import OrganizationConfigurationKey

SECRET_LEAF_NAMES: tuple[str, ...] = (
    "api_key",
    "credentials",
    "aws_access_key",
    "aws_secret_key",
    "aws_session_token",
)

# The ``model_overrides`` sections of a workflow configuration that carry secrets.
MODEL_OVERRIDE_SECTIONS: tuple[str, ...] = ("llm", "tts", "stt", "realtime")

# Secret leaves of a workflow configuration document: the one a workflow and each of its definitions store, and the
# effective document a run freezes from them.
WORKFLOW_CONFIGURATION_SECRET_PATHS: tuple[tuple[str, ...], ...] = tuple(
    [
        ("model_overrides", section, leaf)
        for section in MODEL_OVERRIDE_SECTIONS
        for leaf in SECRET_LEAF_NAMES
    ]
    + [("model_configuration_v2_override", "**", leaf) for leaf in SECRET_LEAF_NAMES]
    + [("voicemail_detection", "api_key")]
    + [("service_tuning", "*", "*", "ctor", "url")]
    # Later PRs append: ("turn", "analyzer", "url")
)

# Columns holding a workflow configuration document. The column name is spelled once: the reads guard of
# test_workflow_configuration_reads_boundary.py matches it quoted, and this module names the column, never reads it.
_WORKFLOW_CONFIGURATIONS = "workflow_configurations"
WORKFLOW_CONFIGURATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("workflow_definitions", _WORKFLOW_CONFIGURATIONS),
    ("workflows", _WORKFLOW_CONFIGURATIONS),
    ("workflow_runs", "effective_configurations"),
)

# Secret fields of workflow graph nodes, at ``nodes[*].data.<field>``: the core QA node and the integration nodes.
NODE_SECRET_FIELDS: tuple[str, ...] = (
    "qa_api_key",
    "noveum_api_key",
    "paygent_api_key",
    "tuner_api_key",
)

# Secret leaves of a workflow graph (lists are transparent in the walk, so ``nodes`` and ``custom_headers`` need no
# ``*``): each node secret field, and every value of the webhook node's custom headers (``[{key, value}]``, sent and
# copied to webhook_deliveries verbatim by api/tasks/run_integrations.py).
WORKFLOW_GRAPH_SECRET_PATHS: tuple[tuple[str, ...], ...] = tuple(
    [("nodes", "data", field) for field in NODE_SECRET_FIELDS]
    + [("nodes", "data", "custom_headers", "value")]
)

# Columns holding a workflow graph: a definition, the copy on the workflow row, and a template.
WORKFLOW_GRAPH_COLUMNS: tuple[tuple[str, str], ...] = (
    ("workflow_definitions", "workflow_json"),
    ("workflows", "workflow_definition"),
    ("workflow_templates", "template_json"),
)


@dataclass(frozen=True)
class SecretSurface:
    table: str
    column: str
    json_path: tuple[str, ...] = ()
    row_key: str | None = None


_ORG_WHOLE_KEYS = (
    OrganizationConfigurationKey.MODEL_CONFIGURATION_V2,  # byok.*
    OrganizationConfigurationKey.TELEPHONY_CONFIGURATION,
    OrganizationConfigurationKey.TWILIO_CONFIGURATION,
    OrganizationConfigurationKey.LANGFUSE_CREDENTIALS,  # secret_key
)

SURFACES: tuple[SecretSurface, ...] = (
    *(
        SecretSurface("organization_configurations", "value", (), key.value)
        for key in _ORG_WHOLE_KEYS
    ),
    # The BigQuery call-event sink keeps its service-account private_key in the org row.
    SecretSurface(
        "organization_configurations",
        "value",
        ("config", "private_key"),
        OrganizationConfigurationKey.CALL_EVENTS.value,
    ),
    SecretSurface("telephony_configurations", "credentials"),
    SecretSurface("external_credentials", "credential_data"),
    SecretSurface("user_configurations", "configuration"),
    SecretSurface("integrations", "connection_details"),
    SecretSurface("webhook_deliveries", "custom_headers"),
    *(
        SecretSurface(table, column, path)
        for table, column in WORKFLOW_CONFIGURATION_COLUMNS
        for path in WORKFLOW_CONFIGURATION_SECRET_PATHS
    ),
    *(
        SecretSurface(table, column, path)
        for table, column in WORKFLOW_GRAPH_COLUMNS
        for path in WORKFLOW_GRAPH_SECRET_PATHS
    ),
    # Static headers of an HTTP API tool and of a transfer tool's resolver (dict[str, str], api/schemas/tool.py).
    SecretSurface("tools", "definition", ("config", "headers", "*")),
    SecretSurface("tools", "definition", ("config", "resolver", "headers", "*")),
)
