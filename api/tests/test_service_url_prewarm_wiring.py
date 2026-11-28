"""Every async caller that builds services awaits the service URL prewarm first (review r2, 394b2ecbec04).

The synchronous builders check each URL on the event loop; without the prewarm each cold host costs a bounded,
logged wait there. The check is on the source: no harness runs all these callers with the guard enforced.
"""

import ast
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
# What checks a tenant service URL synchronously when it is built.
_BUILDERS = frozenset(
    {
        "create_llm_service",
        "create_stt_service",
        "create_tts_service",
        "create_realtime_llm_service",
        "create_llm_service_from_provider",
        "create_llm_service_with_model_override",
        "_create_answer_supervisor",
        "OpenAIEmbeddingService",
        "AzureOpenAIEmbeddingService",
    }
)
_CALLERS = [
    "services/pipecat/run_pipeline.py",
    "services/pipecat/agent_runtime_factory.py",
    "services/workflow/text_chat_runner.py",
    "services/workflow/transfer_introduction.py",
    "services/workflow/qa/llm_config.py",
    "services/gen_ai/embedding/factory.py",
]
_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _own_nodes(function: ast.AsyncFunctionDef):
    """The nodes of this function's body, not those of the functions defined inside it."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, _NESTED)
        )


@pytest.mark.parametrize("module", _CALLERS)
def test_every_async_service_build_awaits_the_prewarm_first(module):
    tree = ast.parse((API_ROOT / module).read_text(encoding="utf-8"))
    checked = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.AsyncFunctionDef):
            continue
        nodes = list(_own_nodes(function))
        builds = [
            node.lineno
            for node in nodes
            if isinstance(node, ast.Call) and _call_name(node) in _BUILDERS
        ]
        if not builds:
            continue
        prewarms = [
            node.lineno
            for node in nodes
            if isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and (_call_name(node.value) or "").startswith("prewarm_")
        ]
        assert prewarms and min(prewarms) < min(builds), (
            f"{module}:{min(builds)} {function.name} builds a service without awaiting the prewarm first"
        )
        checked.append(function.name)
    assert checked, f"{module} no longer builds services in an async function"
