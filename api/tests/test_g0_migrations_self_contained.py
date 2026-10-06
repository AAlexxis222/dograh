"""A migration is a frozen snapshot: it must not import application code, which keeps changing
after the revision ships (Fable review G0 #5)."""

import ast

import pytest

from api.tests._migration_sql import migration_path


@pytest.mark.parametrize("revision", ["5be1d27c9a43", "d7e3a915c2b8"])
def test_migration_does_not_import_application_code(revision):
    tree = ast.parse(migration_path(revision).read_text())
    imported = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not [m for m in imported if m and m.split(".")[0] == "api"], imported
