"""VOZ-N0-24c guard: a registered secret column is written and read only through the credentials-at-rest seam.

The ORM events of api/db/credentials_at_rest.py encrypt on flush and decrypt on load. They do not see SQL that
bypasses the ORM unit of work: a Core ``update``/``insert`` that sets the column, a select of the column itself, or
raw SQL. Each such site must seal (write) or unseal (read) explicitly and is listed in ``HANDLED`` with the reason; a
new one fails this test until it does the same.

The check walks the syntax tree: a Core write of a surface model whose ``values``/``set_`` names a secret column (or
cannot be read: ``**kwargs``, a positional dict, no ``values`` at all); a class-level secret column passed to
``select``/``returning``/``add_columns``/``with_only_columns``; and string literals that UPDATE or INSERT INTO a
surface table or SELECT a secret column FROM it.
"""

import ast
import re
from pathlib import Path

import pytest

from api.db.models import Base
from api.services.security.secret_surfaces import SURFACES

API = Path(__file__).resolve().parents[1]
EXEMPT_DIRS = {"alembic", "tests"}
SELECTORS = {"select", "returning", "add_columns", "with_only_columns"}
WRITERS = {"update", "insert"}
WRITE_SETTERS = {"values", "on_conflict_do_update"}

# (file under api/, enclosing function) -> why it is safe.
HANDLED: dict[tuple[str, str], str] = {
    (
        "db/webhook_credential_client.py",
        "update_credential",
    ): "seals credential_data with the row id before the Core update",
    (
        "db/tool_client.py",
        "update_tool",
    ): "seals definition with the row id before the Core update",
    (
        "db/workflow_client.py",
        "update_definition_configurations",
    ): "seals workflow_configurations with the definition id",
    (
        "db/workflow_client.py",
        "bulk_update_workflow_model_configurations",
    ): "seals each document with its workflow or definition id",
    (
        "db/workflow_client.py",
        "get_definition_configurations_with_owner",
    ): "unseals the selected document with the definition id",
    (
        "db/workflow_client.py",
        "list_draft_definitions_for_backfill",
    ): "unseals each selected document with its definition id",
    (
        "db/workflow_run_client.py",
        "get_workflow_run_config_without_credentials",
    ): "unseals the run snapshot and the pinned document with their ids",
    (
        "db/user_client.py",
        "upsert_user_configuration_value",
    ): "refuses a registered row key (MODEL_CONFIGURATION goes through the ORM)",
    (
        "db/organization_configuration_client.py",
        "claim_configuration_lease",
    ): "bootstrap lease rows (key ORGANIZATION_BOOTSTRAP), never a registered row key",
    (
        "db/organization_configuration_client.py",
        "complete_configuration_lease",
    ): "bootstrap lease rows (key ORGANIZATION_BOOTSTRAP), never a registered row key",
    (
        "db/workflow_recording_client.py",
        "replace_recording_id_in_workflows",
    ): "REPLACE of 'RECORDING_ID: <id>' in the graph text: a token never holds ': ', and row and path stay",
}


def _secret_columns() -> dict[str, tuple[str, set[str]]]:
    """Model class name -> (table, secret column names), from the registry."""
    by_table: dict[str, set[str]] = {}
    for surface in SURFACES:
        by_table.setdefault(surface.table, set()).add(surface.column)
    return {
        mapper.class_.__name__: (
            mapper.local_table.name,
            by_table[mapper.local_table.name],
        )
        for mapper in Base.registry.mappers
        if mapper.local_table.name in by_table
    }


MODELS = _secret_columns()
TABLES = {table: columns for table, columns in MODELS.values()}


def _model_of(node: ast.AST) -> str | None:
    """``Model`` or ``Model.__table__`` -> the model name when it is a surface model."""
    if isinstance(node, ast.Attribute) and node.attr == "__table__":
        node = node.value
    return node.id if isinstance(node, ast.Name) and node.id in MODELS else None


def _func_name(call: ast.Call) -> str | None:
    func = call.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


def _chain(node: ast.AST, parents: dict) -> list[ast.Call]:
    """Calls chained on ``node`` (``update(M).where(..).values(..)``), innermost first."""
    calls = []
    while isinstance(parents.get(node), ast.Attribute):
        attribute = parents[node]
        call = parents.get(attribute)
        if not (isinstance(call, ast.Call) and call.func is attribute):
            break
        calls.append(call)
        node = call
    return calls


def _write_target(call: ast.Call) -> str | None:
    name = _func_name(call)
    if name not in WRITERS:
        return None
    if (
        isinstance(call.func, ast.Attribute) and not call.args
    ):  # Model.__table__.update()
        return _model_of(call.func.value)
    return _model_of(call.args[0]) if call.args else None


def _sets_secret(setter: ast.Call, secrets: set[str]) -> bool:
    if setter.args:
        return True  # a positional dict: cannot tell which columns it sets
    for keyword in setter.keywords:
        if keyword.arg is None or keyword.arg in secrets:
            return True
        if keyword.arg == "set_":
            value = keyword.value
            if not isinstance(value, ast.Dict) or any(
                key is None or not isinstance(key, ast.Constant) or key.value in secrets
                for key in value.keys
            ):
                return True
    return False


def _sql_findings(text: str) -> list[str]:
    found = []
    for table, columns in TABLES.items():
        if not any(re.search(rf"\b{column}\b", text) for column in columns):
            continue
        if re.search(
            rf"\b(?:update|insert\s+into)\s+\"?{table}\b", text, re.IGNORECASE
        ):
            found.append(f"raw SQL writes a secret column of {table}")
        if re.search(rf"\bfrom\s+\"?{table}\b", text, re.IGNORECASE):
            found.append(f"raw SQL selects a secret column of {table}")
    return found


def scan(source: str) -> list[tuple[str, int, str]]:
    """``(enclosing function, line, what)`` of every raw writer or selector of a secret column."""
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def function_of(node):
        while node is not None and not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            node = parents.get(node)
        return node.name if node is not None else "<module>"

    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            model = _write_target(node)
            if model is not None:
                secrets = MODELS[model][1]
                setters = [
                    c for c in _chain(node, parents) if _func_name(c) in WRITE_SETTERS
                ]
                if not setters or any(_sets_secret(s, secrets) for s in setters):
                    findings.append(
                        (function_of(node), node.lineno, f"Core write of {model}")
                    )
            if _func_name(node) in SELECTORS:
                for arg in node.args:
                    if (
                        isinstance(arg, ast.Attribute)
                        and _model_of(arg.value)
                        and arg.attr in MODELS[_model_of(arg.value)][1]
                    ):
                        findings.append(
                            (
                                function_of(node),
                                node.lineno,
                                f"selects {arg.value.id}.{arg.attr}",
                            )
                        )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            findings += [
                (function_of(node), node.lineno, what)
                for what in _sql_findings(node.value)
            ]
    return findings


def _tree_findings() -> list[tuple[str, str, int, str]]:
    found = []
    for path in API.rglob("*.py"):
        relative = path.relative_to(API)
        if relative.parts[0] in EXEMPT_DIRS or "node_modules" in relative.parts:
            continue
        for function, line, what in scan(path.read_text(encoding="utf-8")):
            found.append((relative.as_posix(), function, line, what))
    return found


def test_every_raw_writer_and_selector_of_a_secret_column_is_handled():
    unhandled = [
        f"{path}:{line} in {function}: {what}"
        for path, function, line, what in _tree_findings()
        if (path, function) not in HANDLED
    ]
    assert not unhandled, (
        "seal before a Core write / unseal after a column select (api/db/credentials_at_rest.py), "
        "then list the site in HANDLED:\n" + "\n".join(unhandled)
    )


def test_every_handled_site_still_exists():
    found = {(path, function) for path, function, _line, _what in _tree_findings()}
    assert not set(HANDLED) - found, set(HANDLED) - found


@pytest.mark.parametrize(
    "snippet",
    [
        "def f():\n    update(ToolModel).where(x).values(definition=d)",
        "def f():\n    update(ToolModel).where(x).values(**values)",
        "def f():\n    update(ToolModel).values({'definition': d})",
        "def f():\n    insert(UserConfigurationModel.__table__).values(configuration=v)",
        "def f():\n    insert(UserConfigurationModel).values(key=k).on_conflict_do_update(set_={'configuration': c})",
        "def f():\n    session.execute(insert(WebhookDeliveryModel), rows)",
        "def f():\n    ExternalCredentialModel.__table__.update().values(credential_data=d)",
        "def f():\n    select(WorkflowRunModel.effective_configurations)",
        "def f():\n    select(WorkflowModel.id).add_columns(WorkflowModel.workflow_definition)",
        "def f():\n    stmt.returning(UserConfigurationModel.configuration)",
        "def f():\n    text('UPDATE tools SET definition = :d')",
        "def f():\n    text('INSERT INTO external_credentials (credential_data) VALUES (:d)')",
        "def f():\n    text('SELECT credentials FROM telephony_configurations')",
    ],
)
def test_the_guard_flags_a_raw_access(snippet):
    assert scan(snippet), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        "def f():\n    update(WorkflowDefinitionModel).where(x).values(status='archived', is_current=False)",
        "def f():\n    update(TelephonyConfigurationModel).values(account_id=a)",
        "def f():\n    select(ToolModel).where(ToolModel.organization_id == o)",
        "def f():\n    select(WorkflowModel.id, WorkflowModel.name)",
        "def f():\n    tool.definition = d",
        "def f():\n    text('UPDATE workflow_runs SET state = :s')",
        "def f():\n    text('SELECT id FROM tools')",
    ],
)
def test_the_guard_leaves_innocent_code_alone(snippet):
    assert not scan(snippet), snippet
