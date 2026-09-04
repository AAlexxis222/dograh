"""Architecture guard: workflow configuration is read through the cascade.

After the cascade, runtime code reads the run's frozen effective document
(``run_configurations_for`` / ``get_workflow_run_configurations``). Reading
``.workflow_configurations`` or ``get_definition_configurations(`` anywhere
else lets a draft, the legacy workflow column, or a live organization edit
change a running call. Editor surfaces that manage the stored document are
whitelisted explicitly.
"""

import ast
import re
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
NON_RUNTIME_TOP_LEVEL = {"alembic", "db", "tests"}
STORED_DOCUMENT_SURFACES = {
    Path("routes/workflow.py"),  # GET/PUT/versions/publish of the stored document
    Path("services/configuration/cascade.py"),
    Path(
        "services/configuration/ai_model_configuration.py"
    ),  # v2 sweep rewrites stored docs
    Path("services/workflow/duplicate.py"),  # copies the stored sparse document (F5)
    Path("services/workflow/run_creation.py"),
    Path(
        "services/workflow/configuration_policy.py"
    ),  # reads stored draft/published to build the effective
    Path("conftest.py"),
}
# Negative lookbehind: ``api.schemas.workflow_configurations`` imports are not reads.
READ_PATTERN = re.compile(
    r"(?<!schemas)\.workflow_configurations\b|get_definition_configurations\("
)


def _runtime_files():
    for path in API_ROOT.rglob("*.py"):
        relative = path.relative_to(API_ROOT)
        if (
            relative.parts[0] in NON_RUNTIME_TOP_LEVEL
            or relative in STORED_DOCUMENT_SURFACES
        ):
            continue
        if "__pycache__" in relative.parts:
            continue
        yield path, relative


def test_runtime_reads_workflow_configuration_only_through_the_cascade():
    violations = [
        f"{relative}:{number}: {line.strip()}"
        for path, relative in _runtime_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if READ_PATTERN.search(line) and not line.lstrip().startswith("#")
    ]
    assert violations == [], (
        "read the run's frozen configuration (run_configurations_for / "
        f"get_workflow_run_configurations) instead: {violations}"
    )


def test_every_run_creator_freezes_the_effective_configuration():
    missing = []
    for path, relative in _runtime_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_workflow_run"
                and not any(
                    kw.arg == "effective_configurations" for kw in node.keywords
                )
            ):
                missing.append(f"{relative}:{node.lineno}")
    assert missing == [], (
        f"create_workflow_run without effective_configurations=: {missing}"
    )
