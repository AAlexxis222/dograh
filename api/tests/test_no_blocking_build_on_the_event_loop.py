"""No async function runs a service build or a service URL check on the event loop (review r3, RUL-r3-12 (5)).

The builders are synchronous and check every tenant URL they connect to; on an event loop a host the guard's cache
has not resolved raises ServiceHostUnresolved. Async code therefore hands a builder to
``build_resolving_service_hosts`` (or, for the configuration validator, to ``asyncio.to_thread``) instead of calling
it. The check is on the source, over every module of the api, so a new caller is covered without a list of callers.
"""

import ast
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
# What checks a tenant service URL synchronously.
_SEED = frozenset(
    {
        "create_llm_service",
        "create_stt_service",
        "create_tts_service",
        "create_realtime_llm_service",
        "create_llm_service_from_provider",
        "create_llm_service_with_model_override",
        "check_service_url",
        "OpenAIEmbeddingService",
        "AzureOpenAIEmbeddingService",
        "DograhEmbeddingService",
    }
)
# The calls that may receive a member as an argument: they run it off the loop, or resolve its hosts off the loop.
_RUNNERS = frozenset({"build_resolving_service_hosts", "to_thread"})
_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _own_nodes(function):
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


def _functions():
    """(module, function, enclosing class name or None) for every def under api/ outside the tests."""
    for path in sorted(API_ROOT.rglob("*.py")):
        module = path.relative_to(API_ROOT).as_posix()
        if module.startswith(("tests/", "alembic/")) or "/tests/" in module:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    owner = node.name if isinstance(node, ast.ClassDef) else None
                    yield module, child, owner


def _members(functions) -> set[str]:
    """The seed, closed over every sync def whose own body calls a member (a constructor counts as its class)."""
    members = set(_SEED)
    changed = True
    while changed:
        changed = False
        for _, function, owner in functions:
            if not isinstance(function, ast.FunctionDef):
                continue
            calls = (
                _name(node.func)
                for node in _own_nodes(function)
                if isinstance(node, ast.Call)
            )
            if any(call in members for call in calls):
                name = owner if function.name == "__init__" and owner else function.name
                if name not in members:
                    members.add(name)
                    changed = True
    return members


def _violations(functions, members):
    for module, function, _ in functions:
        if not isinstance(function, ast.AsyncFunctionDef):
            continue
        for node in _own_nodes(function):
            if not isinstance(node, ast.Call):
                continue
            called = _name(node.func)
            if called in members:
                yield f"{module}:{node.lineno} {function.name} calls {called} on the event loop"
            elif called not in _RUNNERS:
                for arg in node.args:
                    if _name(arg) in members:
                        yield (
                            f"{module}:{node.lineno} {function.name} hands {_name(arg)} to {called}, "
                            "which does not resolve its hosts off the loop"
                        )


def test_no_async_function_builds_a_service_on_the_event_loop():
    functions = list(_functions())
    members = _members(functions)
    # The closure reaches the builders behind the seed (a guard against a rule that checks nothing).
    assert {
        "_create_answer_supervisor",
        "_validate_service",
        "_validate_configuration",
        "_validate_runtime_service_url",
    } <= members
    violations = list(_violations(functions, members))
    assert not violations, "\n".join(violations)


def test_async_callers_hand_their_builds_to_the_wrapper():
    wrapped = {
        module
        for module, function, _ in _functions()
        if isinstance(function, ast.AsyncFunctionDef)
        for node in _own_nodes(function)
        if isinstance(node, ast.Call)
        and _name(node.func) == "build_resolving_service_hosts"
    }
    assert {
        "services/pipecat/run_pipeline.py",
        "services/pipecat/agent_runtime_factory.py",
        "services/workflow/text_chat_runner.py",
        "services/workflow/transfer_introduction.py",
        "services/workflow/qa/llm_config.py",
        "services/gen_ai/embedding/factory.py",
    } <= wrapped
