"""Architecture guard: workflow configuration is read through the cascade.

After the cascade, runtime code reads the run's frozen effective document
(``run_configurations_for`` / ``get_workflow_run_configurations``). Reading
``.workflow_configurations`` (however it is spelled) or
``get_definition_configurations_with_owner(`` anywhere
else lets a draft, the legacy workflow column, or a live organization edit
change a running call. Editor surfaces that manage the stored document are
whitelisted explicitly.

Known blind spots of the creator guard, both matched by name at the call site:
a creator reached through an alias or a ``functools.partial`` is invisible to
it, and so is one added inside ``api/db/`` (skipped as non-runtime). Every
creator in the tree today calls ``create_workflow_run`` directly from a service
or a route.
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
    Path("services/workflow/duplicate.py"),  # copies the stored sparse document
    Path("services/workflow/run_creation.py"),
    Path(
        "services/workflow/configuration_policy.py"
    ),  # reads stored draft/published to build the effective
    Path("conftest.py"),
}
# Negative lookbehind: ``api.schemas.workflow_configurations`` imports are not reads.
READ_PATTERN = re.compile(
    r"(?<!schemas)\.workflow_configurations\b"
    # The same read spelled dynamically: getattr(x, "workflow_configurations")
    # or x["workflow_configurations"], in either quote style.
    r"|[\[(,]\s*[\"']workflow_configurations[\"']"
    r"|get_definition_configurations_with_owner\("
)


def _python_files(*, exclude_stored_document_surfaces: bool):
    """Yield every runtime .py file under the API root.

    Both guards skip the non-runtime top-level dirs (migrations, the db
    layer, and the test suite itself) and __pycache__. Only the read guard
    also skips STORED_DOCUMENT_SURFACES: those files legitimately read the
    stored document to build or edit it. The creator guard must NOT use that
    whitelist — routes/workflow.py, run_creation.py and conftest.py all
    contain (or could contain) a `create_workflow_run(` call that must still
    be checked.
    """
    for path in API_ROOT.rglob("*.py"):
        relative = path.relative_to(API_ROOT)
        if relative.parts[0] in NON_RUNTIME_TOP_LEVEL:
            continue
        if "__pycache__" in relative.parts:
            continue
        if exclude_stored_document_surfaces and relative in STORED_DOCUMENT_SURFACES:
            continue
        yield path, relative


def test_runtime_reads_workflow_configuration_only_through_the_cascade():
    violations = [
        f"{relative}:{number}: {line.strip()}"
        for path, relative in _python_files(exclude_stored_document_surfaces=True)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if READ_PATTERN.search(line) and not line.lstrip().startswith("#")
    ]
    assert violations == [], (
        "read the run's frozen configuration (run_configurations_for / "
        f"get_workflow_run_configurations) instead: {violations}"
    )


def test_every_run_creator_freezes_the_effective_configuration():
    missing = []
    for path, relative in _python_files(exclude_stored_document_surfaces=False):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                callee_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                callee_name = node.func.id
            else:
                continue
            if callee_name != "create_workflow_run":
                continue
            if not any(kw.arg == "effective_configurations" for kw in node.keywords):
                missing.append(f"{relative}:{node.lineno}")
    assert missing == [], (
        f"create_workflow_run without effective_configurations=: {missing}"
    )
