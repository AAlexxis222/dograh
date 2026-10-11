"""VOZ-N0-24a guard: nothing queries inside `telephony_configurations.credentials` in SQL.

The column is encrypted whole by a later task, after which a JSON path into it matches nothing.
What the SQL needs lives in plaintext columns (`account_id`); anything else is read in Python
from the loaded row. This test fails the day a query reaches into the column again.

The check walks the syntax tree, so it does not care how a statement is spelled or split over
lines: it flags the class-level column (also through an alias, `getattr` or `__table__.c`),
SQLAlchemy JSON accessors applied to a `credentials` attribute, `literal_column("credentials")`,
and any string literal that applies a JSON operator or `json*_` function to `credentials`.
An instance's `.credentials` (a plain dict) is read freely.
"""

import ast
import re
from pathlib import Path

API = Path(__file__).resolve().parents[1]
MODEL = "TelephonyConfigurationModel"

# Applied to string literals (raw SQL), operators after an optional closing paren and cast.
SQL_OPERATOR = re.compile(
    r"credentials[\s)\"']*(?:::\w+[\s)]*)?(?:->>?|#>>?|@>)",
    re.IGNORECASE,
)
# Any cast of the column (`credentials::jsonb ? :f`, `credentials::text LIKE :p`) or a
# CAST(credentials AS ...): the column is only cast to be searched.
SQL_CAST = re.compile(
    r"credentials[\s)\"']*::\w+|\bCAST\s*\(\s*\(?\s*credentials\b", re.IGNORECASE
)
SQL_FUNCTION = re.compile(r"\bjsonb?_\w+\s*\(\s*(?:\w+\.)?\"?credentials\b")
SQL_PATTERNS = (SQL_OPERATOR, SQL_CAST, SQL_FUNCTION)
TABLE_COLLECTIONS = {"c", "columns"}
# SQLAlchemy accessors that only exist on a JSON column expression, never on a dict.
JSON_ACCESSORS = {
    "astext",
    "op",
    "as_string",
    "as_integer",
    "as_float",
    "as_numeric",
    "as_boolean",
    "as_json",
    "has_key",
    "has_all",
    "has_any",
    "contains",
    "contained_by",
    "cast",
    "label",
    "in_",
}
EXEMPT_DIRS = {"alembic", "tests"}


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_credentials_expression(node: ast.AST) -> bool:
    """`<x>.credentials`, optionally subscripted: `<x>.credentials["field"]`."""
    if isinstance(node, ast.Subscript):
        node = node.value
    return isinstance(node, ast.Attribute) and node.attr == "credentials"


def _is_credentials_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value == "credentials"


def offences(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == MODEL:
                if node.attr == "credentials":
                    found.append(f"{line}: {MODEL}.credentials")
            if node.attr == "credentials" and _name(node.value) in TABLE_COLLECTIONS:
                found.append(f"{line}: table.c.credentials")
            if node.attr in JSON_ACCESSORS and _is_credentials_expression(node.value):
                found.append(f"{line}: JSON accessor .{node.attr} on credentials")
        elif isinstance(node, ast.Subscript):
            if _name(node.value) in TABLE_COLLECTIONS and _is_credentials_literal(
                node.slice
            ):
                found.append(f"{line}: table.c['credentials']")
        elif isinstance(node, ast.Call):
            func = _name(node.func)
            args = node.args
            if func in {"literal_column", "column"} and _is_credentials_literal(
                args[0] if args else None
            ):
                found.append(f"{line}: {func}('credentials')")
            if func == "aliased" and args and _name(args[0]) == MODEL:
                found.append(f"{line}: aliased({MODEL})")
            if func == "getattr" and len(args) > 1 and _name(args[0]) == MODEL:
                found.append(f"{line}: getattr({MODEL}, ...)")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(pattern.search(node.value) for pattern in SQL_PATTERNS):
                found.append(
                    f"{line}: SQL string applies a JSON operator to credentials"
                )
    return found


def test_no_code_queries_inside_telephony_credentials():
    found = {}
    for path in sorted(API.rglob("*.py")):
        relative = path.relative_to(API)
        if relative.parts[0] in EXEMPT_DIRS or "venv" in relative.parts[0]:
            continue
        hits = offences(path.read_text(encoding="utf-8"))
        if hits:
            found[relative.as_posix()] = hits
    assert not found, (
        "SQL reaches into telephony_configurations.credentials; use the account_id column "
        f"or read the loaded row in Python: {found}"
    )


OLD_FORMS = {
    "class attribute operator": (
        'TelephonyConfigurationModel.credentials.op("->>")(field) == account_id'
    ),
    "text json operator": (
        "sa.text(\"SELECT credentials->>'account_sid' FROM telephony_configurations\")"
    ),
    "table column": "select(TelephonyConfigurationModel.__table__.c.credentials)",
}
REVIEW_FORMS = {
    "aliased + astext": (
        "cfg = aliased(TelephonyConfigurationModel)\n"
        "stmt = select(cfg).where(cfg.credentials['account_sid'].astext == a)\n"
    ),
    "accessor on an unknown alias": 'select(1).where(row.credentials["account_sid"].astext == a)',
    "operator on an unknown alias": 'select(1).where(row.credentials.op("->>")(f) == a)',
    "aliased alone": "cfg = aliased(TelephonyConfigurationModel)",
    "getattr": 'getattr(TelephonyConfigurationModel, "credentials")',
    "text with a cast": (
        'text("SELECT 1 FROM telephony_configurations WHERE credentials::jsonb ->> :f = :v")'
    ),
    "table.c subscript": "select(TelephonyConfigurationModel.__table__.c['credentials'])",
    "literal_column": 'literal_column("credentials").op("->>")(field)',
    "parenthesised column": (
        'text("SELECT 1 FROM telephony_configurations WHERE (credentials) ->> :f = :v")'
    ),
    "json function": 'text("SELECT json_extract_path_text(credentials, :f) FROM t")',
    "chain split over lines": (
        "select(TelephonyConfigurationModel\n    .credentials\n    .op('->>')(f))"
    ),
    "string split over lines": (
        'text("SELECT 1 FROM telephony_configurations "\n     "WHERE credentials "\n'
        '     "->> :f = :v")'
    ),
}
ROUND2_FORMS = {
    "CAST(credentials AS jsonb)": (
        'text("SELECT 1 FROM telephony_configurations WHERE CAST(credentials AS jsonb) ->> :f = :v")'
    ),
    "jsonb existence operator": (
        'text("SELECT 1 FROM telephony_configurations WHERE credentials::jsonb ? :f")'
    ),
    "text cast with LIKE": (
        'text("SELECT 1 FROM telephony_configurations WHERE credentials::text LIKE :p")'
    ),
    "quoted identifier in a JSON function": (
        'text("SELECT json_extract_path_text(\\"credentials\\", :f) FROM t")'
    ),
    "table.columns subscript": "select(TelephonyConfigurationModel.__table__.columns['credentials'])",
    "table.columns attribute": "select(TelephonyConfigurationModel.__table__.columns.credentials)",
}
INNOCENT = {
    "instance read": "account_id = (config.credentials or {}).get('x')",
    "instance subscript": "value = config.credentials['api_key']",
    "annotation arrow": "def fetch(credentials) -> dict:\n    return {}\n",
    "docstring": 'def f():\n    """Reads the credentials of the account."""\n',
    "other dict key": "payload = {'credentials': secret}",
    "prose question": 'raise ValueError("Missing credentials? Add them first.")',
}


def test_the_guard_recognises_every_known_spelling():
    for name, source in {**OLD_FORMS, **REVIEW_FORMS, **ROUND2_FORMS}.items():
        assert offences(source), name


def test_the_guard_leaves_python_reads_alone():
    for name, source in INNOCENT.items():
        assert not offences(source), name
