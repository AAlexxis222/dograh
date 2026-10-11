"""VOZ-N0-24c guard: a registered secret column is written and read only through the credentials-at-rest seam.

The ORM events of api/db/credentials_at_rest.py encrypt on flush and decrypt on load. They do not see SQL that
bypasses the ORM unit of work: a Core or bulk write that sets the column, a read of the column itself, a SQL filter
into it (it holds a token, so a JSON path matches nothing: the 24a inbound-routing outage), or raw SQL. Each such
site must seal (write) or unseal (read) explicitly and is listed in ``HANDLED`` with the reason; a new one fails this
test until it does the same. Python files under api/ (except alembic and tests) and scripts/ are scanned.

What the syntax-tree walk flags:
- any class-level reference to a secret column: ``Model.col``, ``Model.__table__.c.col`` / ``.c["col"]``,
  ``getattr(Model, "col")``, the same through ``X = aliased(Model)``; loader options (``load_only``, ``defer``, ...)
  are ORM and allowed;
- ``Model.__table__`` passed to ``select`` and ``Model.__table__.select()``;
- a write through any callee named ``*insert``/``*update`` (``pg_insert``, ``sa.update``, ``Model.__table__.update()``,
  legacy ``query(Model)...update(...)``) whose values name a secret column or cannot be read, and the bulk
  ``bulk_insert_mappings``/``bulk_update_mappings``/``bulk_save_objects`` (mapper events never fire);
- string literals (f-strings included) that reach a surface table (FROM/JOIN/UPDATE/INTO) and name a secret column,
  select ``*``, return ``*`` or build JSON from the row (``row_to_json``, ``to_json``, ``json_agg``).
"""

import ast
import re
from pathlib import Path

import pytest

from api.db.models import Base
from api.services.security.secret_surfaces import SURFACES

REPO = Path(__file__).resolve().parents[2]
ROOTS = (REPO / "api", REPO / "scripts")
EXEMPT_DIRS = {("api", "alembic"), ("api", "tests")}
SELECTORS = {"select", "returning", "add_columns", "with_only_columns", "column"}
WRITE_SETTERS = {"values", "on_conflict_do_update", "update"}
BULK = {"bulk_insert_mappings", "bulk_update_mappings", "bulk_save_objects"}
LOADER_OPTIONS = {
    "load_only",
    "defer",
    "undefer",
    "deferred",
    "selectinload",
    "joinedload",
    "subqueryload",
    "lazyload",
    "raiseload",
    "noload",
    "defaultload",
}

# (file under the repo, enclosing function) -> why it is safe.
HANDLED: dict[tuple[str, str], str] = {
    (
        "api/db/webhook_credential_client.py",
        "update_credential",
    ): "seals credential_data with the row id before the Core update",
    (
        "api/db/tool_client.py",
        "update_tool",
    ): "seals definition with the row id before the Core update",
    (
        "api/db/workflow_client.py",
        "update_definition_configurations",
    ): "seals workflow_configurations with the definition id",
    (
        "api/db/workflow_client.py",
        "bulk_update_workflow_model_configurations",
    ): "seals each document with its workflow or definition id",
    (
        "api/db/workflow_client.py",
        "get_definition_configurations_with_owner",
    ): "unseals the selected document with the definition id",
    (
        "api/db/workflow_client.py",
        "list_draft_definitions_for_backfill",
    ): "unseals each selected document with its definition id",
    (
        "api/db/workflow_run_client.py",
        "get_workflow_run_config_without_credentials",
    ): "unseals the run snapshot and the pinned document with their ids",
    (
        "api/db/user_client.py",
        "upsert_user_configuration_value",
    ): "refuses a registered row key (MODEL_CONFIGURATION goes through the ORM)",
    (
        "api/db/organization_configuration_client.py",
        "claim_configuration_lease",
    ): "bootstrap lease rows (key ORGANIZATION_BOOTSTRAP), never a registered row key",
    (
        "api/db/organization_configuration_client.py",
        "complete_configuration_lease",
    ): "bootstrap lease rows (key ORGANIZATION_BOOTSTRAP), never a registered row key",
    (
        "api/db/organization_configuration_client.py",
        "release_configuration_lease",
    ): "bootstrap lease rows (key ORGANIZATION_BOOTSTRAP), never a registered row key",
    (
        "api/db/workflow_recording_client.py",
        "replace_recording_id_in_workflows",
    ): "REPLACE of 'RECORDING_ID: <id>' in the graph text: a token never holds ': ', and row and path stay",
    (
        "scripts/xpand/deprecation_sweep.py",
        "_load_rows",
    ): "open_encrypted_rows opens every whole-column token; a token it cannot open exits 3, never a clean 0",
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
TABLE_REF = r"\b(?:from|join|update|into)\s+\"?({tables})\b"
SQL_TABLE = re.compile(
    TABLE_REF.format(tables="|".join(map(re.escape, TABLES))), re.IGNORECASE
)
# An f-string whose table name is interpolated (NUL marks an interpolation): the walk cannot tell which table.
SQL_DYNAMIC_TABLE = re.compile(
    r"\bselect\b.*?\b(?:from|join)\s+\"?\x00|\bupdate\s+\"?\x00\"?\s+set\b|\binsert\s+into\s+\"?\x00",
    re.IGNORECASE | re.DOTALL,
)
SQL_WHOLE_ROW = re.compile(
    r"\bselect\s+(?:distinct\s+)?(?:\w+\.)?\*|\breturning\s+(?:\w+\.)?\*|\brow_to_json\b"
    r"|\b(?:to_jsonb?|jsonb?_agg)\s*\(\s*\w+\s*\)",
    re.IGNORECASE,
)


def _func_name(call: ast.Call) -> str | None:
    func = call.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


class _Scan:
    def __init__(self, source: str):
        self.tree = ast.parse(source)
        self.parents = {
            child: parent
            for parent in ast.walk(self.tree)
            for child in ast.iter_child_nodes(parent)
        }
        # Names bound to ``aliased(SurfaceModel)`` stand for that model.
        self.models = {name: name for name in MODELS}
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and _func_name(node.value) == "aliased"
                and node.value.args
                and self.model_of(node.value.args[0])
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.models[target.id] = self.model_of(node.value.args[0])
        self.findings: list[tuple[str, int, str]] = []

    def model_of(self, node: ast.AST, *, table_ok: bool = False) -> str | None:
        """``Model`` (or an alias of it); with ``table_ok`` also ``Model.__table__``."""
        if table_ok and isinstance(node, ast.Attribute) and node.attr == "__table__":
            node = node.value
        return self.models.get(node.id) if isinstance(node, ast.Name) else None

    def function_of(self, node) -> str:
        while node is not None and not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            node = self.parents.get(node)
        return node.name if node is not None else "<module>"

    def flag(self, node, what: str) -> None:
        self.findings.append((self.function_of(node), node.lineno, what))

    def in_loader_option(self, node) -> bool:
        while node is not None:
            node = self.parents.get(node)
            if isinstance(node, ast.Call) and _func_name(node) in LOADER_OPTIONS:
                return True
        return False

    def chain(self, node) -> list[ast.Call]:
        """Calls chained on ``node`` (``update(M).where(..).values(..)``), innermost first."""
        calls = []
        while isinstance(self.parents.get(node), ast.Attribute):
            attribute = self.parents[node]
            call = self.parents.get(attribute)
            if not (isinstance(call, ast.Call) and call.func is attribute):
                break
            calls.append(call)
            node = call
        return calls

    def secret_reference(self, node: ast.AST) -> str | None:
        """``Model.col``, ``Model.__table__.c.col``, ``Model.__table__.c["col"]`` -> "Model.col"."""
        if isinstance(node, ast.Attribute):
            model = self.model_of(node.value)
            if model and node.attr in MODELS[model][1]:
                return f"{model}.{node.attr}"
            if isinstance(node.value, ast.Attribute) and node.value.attr in (
                "c",
                "columns",
            ):
                model = self.model_of(node.value.value, table_ok=True)
                if model and node.attr in MODELS[model][1]:
                    return f"{model}.{node.attr}"
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr in ("c", "columns")
            and isinstance(node.slice, ast.Constant)
        ):
            model = self.model_of(node.value.value, table_ok=True)
            if model and node.slice.value in MODELS[model][1]:
                return f"{model}.{node.slice.value}"
        return None

    def query_model(self, node: ast.AST) -> str | None:
        """The surface model of a ``query(Model)`` somewhere down a method chain."""
        while isinstance(node, (ast.Call, ast.Attribute)):
            if isinstance(node, ast.Call):
                if (
                    _func_name(node) == "query"
                    and node.args
                    and self.model_of(node.args[0])
                ):
                    return self.model_of(node.args[0])
                node = node.func
            else:
                node = node.value
        return None

    def write_target(self, call: ast.Call) -> str | None:
        name = _func_name(call) or ""
        if not name.endswith(("insert", "update")):
            return None
        if isinstance(call.func, ast.Attribute):
            if (
                call.func.attr in ("update", "insert") and not call.args
            ):  # Model.__table__.update()
                return self.model_of(call.func.value, table_ok=True)
            if call.func.attr == "update" and (
                model := self.query_model(call.func.value)
            ):
                return model  # legacy Query.update({...})
        return self.model_of(call.args[0], table_ok=True) if call.args else None

    def sets_secret(self, setter: ast.Call, secrets: set[str]) -> bool:
        def secret_key(key) -> bool:
            if isinstance(key, ast.Constant):
                return key.value in secrets
            return (
                key is None
                or self.secret_reference(key) is not None
                or not isinstance(key, ast.Attribute)
            )

        for arg in setter.args:
            if not isinstance(arg, ast.Dict) or any(
                secret_key(key) for key in arg.keys
            ):
                return True
        for keyword in setter.keywords:
            if keyword.arg is None or keyword.arg in secrets:
                return True
            if keyword.arg == "set_":
                value = keyword.value
                if not isinstance(value, ast.Dict) or any(
                    secret_key(key) for key in value.keys
                ):
                    return True
        return False

    def call(self, node: ast.Call) -> None:
        name = _func_name(node)
        model = self.write_target(node)
        if model is not None:
            secrets = MODELS[model][1]
            if _func_name(node) == "update" and self.query_model(node.func):
                setters = [node]
            else:
                setters = [
                    c for c in self.chain(node) if _func_name(c) in WRITE_SETTERS
                ]
            if not setters or any(self.sets_secret(s, secrets) for s in setters):
                self.flag(node, f"write of {model}")
        if name in BULK:
            self.flag(node, f"bulk write ({name})")
        if name in SELECTORS:
            for arg in node.args:
                if self.model_of(arg) is None and (
                    model := self.model_of(arg, table_ok=True)
                ):
                    self.flag(node, f"selects the whole table of {model}")
        if (
            name == "select"
            and isinstance(node.func, ast.Attribute)
            and (model := self.model_of(node.func.value, table_ok=True))
            and self.model_of(node.func.value) is None
        ):
            self.flag(node, f"selects the whole table of {model}")
        if (
            name == "getattr"
            and len(node.args) >= 2
            and (model := self.model_of(node.args[0]))
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in MODELS[model][1]
        ):
            self.flag(node, f"references {model}.{node.args[1].value}")

    def string(self, node, text: str) -> None:
        tables = {match.lower() for match in SQL_TABLE.findall(text)}
        for table in sorted(tables):
            if SQL_WHOLE_ROW.search(text) or any(
                re.search(rf"\b{c}\b", text) for c in TABLES[table]
            ):
                self.flag(node, f"raw SQL reaches a secret column of {table}")

    def run(self) -> list[tuple[str, int, str]]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                self.call(node)
            elif isinstance(node, (ast.Attribute, ast.Subscript)):
                reference = self.secret_reference(node)
                outer = self.parents.get(node)
                nested = isinstance(
                    outer, (ast.Attribute, ast.Subscript)
                ) and self.secret_reference(outer)
                if reference and not nested and not self.in_loader_option(node):
                    self.flag(node, f"references {reference}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if not isinstance(self.parents.get(node), ast.JoinedStr):
                    self.string(node, node.value)
            elif isinstance(node, ast.JoinedStr):
                text = "".join(
                    part.value if isinstance(part, ast.Constant) else "\x00"
                    for part in node.values
                )
                if SQL_DYNAMIC_TABLE.search(text):
                    self.flag(node, "raw SQL on an interpolated table name")
                self.string(node, text)
        return self.findings


def scan(source: str) -> list[tuple[str, int, str]]:
    """``(enclosing function, line, what)`` of every raw access to a secret column."""
    return _Scan(source).run()


def _tree_findings() -> list[tuple[str, str, int, str]]:
    found = []
    for root in ROOTS:
        for path in root.rglob("*.py"):
            relative = path.relative_to(REPO)
            if relative.parts[:2] in EXEMPT_DIRS or "node_modules" in relative.parts:
                continue
            for function, line, what in scan(path.read_text(encoding="utf-8")):
                found.append((relative.as_posix(), function, line, what))
    return found


def test_every_raw_access_to_a_secret_column_is_handled():
    unhandled = [
        f"{path}:{line} in {function}: {what}"
        for path, function, line, what in _tree_findings()
        if (path, function) not in HANDLED
    ]
    assert not unhandled, (
        "seal before a raw write / unseal after a raw read (api/db/credentials_at_rest.py), never filter "
        "inside the column in SQL, then list the site in HANDLED:\n"
        + "\n".join(unhandled)
    )


def test_every_handled_site_still_exists():
    found = {(path, function) for path, function, _line, _what in _tree_findings()}
    assert not set(HANDLED) - found, set(HANDLED) - found


@pytest.mark.parametrize(
    "snippet",
    [
        # Core and bulk writes
        "def f():\n    update(ToolModel).where(x).values(definition=d)",
        "def f():\n    update(ToolModel).where(x).values(**values)",
        "def f():\n    update(ToolModel).values({'definition': d})",
        "def f():\n    insert(UserConfigurationModel.__table__).values(configuration=v)",
        "def f():\n    insert(UserConfigurationModel).values(key=k).on_conflict_do_update(set_={'configuration': c})",
        "def f():\n    session.execute(insert(WebhookDeliveryModel), rows)",
        "def f():\n    ExternalCredentialModel.__table__.update().values(credential_data=d)",
        "def f():\n    pg_insert(ToolModel).values(definition=d)",
        "def f():\n    sa.update(ToolModel).values(definition=d)",
        "def f():\n    session.query(ToolModel).filter(x).update({'definition': d})",
        "def f():\n    session.query(ToolModel).filter(x).update({ToolModel.definition: d})",
        "def f():\n    session.bulk_insert_mappings(ToolModel, rows)",
        "def f():\n    session.bulk_update_mappings(ToolModel, rows)",
        "def f():\n    session.bulk_save_objects(rows)",
        "def f():\n    insert(ToolModel).from_select(['definition'], q)",
        # column reads and SQL filters inside the column
        "def f():\n    select(WorkflowRunModel.effective_configurations)",
        "def f():\n    select(WorkflowModel.id).add_columns(WorkflowModel.workflow_definition)",
        "def f():\n    stmt.returning(UserConfigurationModel.configuration)",
        "def f():\n    select(TelephonyConfigurationModel).where(TelephonyConfigurationModel.credentials['account_sid'].astext == x)",
        "def f():\n    select(ToolModel.definition['config'])",
        "def f():\n    select(func.coalesce(ToolModel.definition, x))",
        "def f():\n    select(cast(ToolModel.definition, JSONB))",
        "def f():\n    select(ToolModel.__table__.c.definition)",
        "def f():\n    select(ToolModel.__table__.c['definition'])",
        "def f():\n    select(ToolModel.__table__)",
        "def f():\n    ToolModel.__table__.select()",
        "def f():\n    getattr(ToolModel, 'definition')",
        "T = aliased(ToolModel)\ndef f():\n    select(T.definition)",
        # raw SQL
        "def f():\n    text('UPDATE tools SET definition = :d')",
        "def f():\n    text('INSERT INTO external_credentials (credential_data) VALUES (:d)')",
        "def f():\n    text('SELECT credentials FROM telephony_configurations')",
        "def f():\n    text('SELECT * FROM tools')",
        "def f():\n    text('SELECT t.* FROM tools t')",
        "def f():\n    text('SELECT row_to_json(t) FROM telephony_configurations t')",
        "def f():\n    text('UPDATE tools SET status = :s RETURNING *')",
        "def f():\n    text(f'SELECT id, value FROM organization_configurations WHERE id = {i}')",
        "def f():\n    text(f'SELECT id, {s.column} FROM {s.table} {s.where}')",
        "def f():\n    text('SELECT json_agg(t) FROM tools t')",
    ],
)
def test_the_guard_flags_a_raw_access(snippet):
    assert scan(snippet), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        "def f():\n    update(WorkflowDefinitionModel).where(x).values(status='archived', is_current=False)",
        "def f():\n    update(TelephonyConfigurationModel).values(account_id=a)",
        "def f():\n    session.query(ToolModel).filter(x).update({'status': s})",
        "def f():\n    select(ToolModel).where(ToolModel.organization_id == o)",
        "def f():\n    select(WorkflowModel.id, WorkflowModel.name)",
        "def f():\n    select(ToolModel).options(load_only(ToolModel.id), defer(ToolModel.definition))",
        "def f():\n    tool.definition = d",
        "def f():\n    config.value.get('x')",
        "def f():\n    ToolModel.__table__.delete().where(x)",
        "def f():\n    text('UPDATE workflow_runs SET state = :s')",
        "def f():\n    text('SELECT id FROM tools')",
        "def f():\n    text('Pick the tools from the list')",
        "def f():\n    text('UPDATE workflow_runs SET gathered_context = to_jsonb(cast(:u as text))')",
    ],
)
def test_the_guard_leaves_innocent_code_alone(snippet):
    assert not scan(snippet), snippet
