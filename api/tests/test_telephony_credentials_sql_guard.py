"""VOZ-N0-24a guard: nothing queries inside `telephony_configurations.credentials` in SQL.

The column is encrypted whole by a later task, after which a JSON path into it matches nothing.
What the SQL needs lives in plaintext columns (`account_id`); anything else is read in Python
from the loaded row. This test fails the day a query reaches into the column again.
"""

import re
from pathlib import Path

API = Path(__file__).resolve().parents[1]

# The class-level attribute (a SQL expression; an instance's `.credentials` is a plain dict)
# and any JSON operator on a column or table named `credentials`.
SQL_REACH = re.compile(
    r"TelephonyConfigurationModel\s*\.\s*credentials\b"
    r"|credentials\s*(?:->>?|#>>?)"
    r"|\.c\.credentials\b"
)
EXEMPT_DIRS = {"alembic", "tests"}


def offending_lines(source: str) -> list[str]:
    return [
        f"{number}: {line.strip()}"
        for number, line in enumerate(source.splitlines(), 1)
        if SQL_REACH.search(line)
    ]


def test_no_code_queries_inside_telephony_credentials():
    found = {}
    for path in sorted(API.rglob("*.py")):
        relative = path.relative_to(API)
        if relative.parts[0] in EXEMPT_DIRS or "venv" in relative.parts[0]:
            continue
        lines = offending_lines(path.read_text(encoding="utf-8"))
        if lines:
            found[relative.as_posix()] = lines
    assert not found, (
        "SQL reaches into telephony_configurations.credentials; use the account_id column "
        f"or read the loaded row in Python: {found}"
    )


def test_the_guard_recognises_the_old_queries():
    for line in (
        'TelephonyConfigurationModel.credentials.op("->>")(field) == account_id',
        "sa.text(\"SELECT credentials->>'account_sid' FROM telephony_configurations\")",
        "select(TelephonyConfigurationModel.__table__.c.credentials)",
    ):
        assert offending_lines(line), line
    assert not offending_lines("account_id = (config.credentials or {}).get('x')")
