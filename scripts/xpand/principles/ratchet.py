#!/usr/bin/env python3
"""Principles ratchet (review system v3, spec 2026-10-06 section 3).

A finding blocks when its value at the PR head is above the limit AND above every reference tree (the PR base and,
in the fork, the merge-base with upstream). Function rules: value = metric of that function (file, name).
Count rules: value = number of findings of that rule in that file, limit 0. New import cycles block.
Everything above the limit that did not get worse is reported as debt. Exit 0 pass, 1 blocked, 2 tool/git error.
source: xpand plugin tools/principles @c0477d6
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

# The linter configs next to this file use the same numbers (test_configs_match_limits).
FUNCTION_LIMITS = {
    "gocyclo": 15, "C901": 15, "complexity": 15,
    "funlen": 80, "max-lines-per-function": 80, "PLR0915": 50,
}


class ToolError(Exception):
    """A linter or git failed: the gate fails closed."""


@dataclass(frozen=True)
class Finding:
    file: str
    rule: str
    name: str
    value: int


QUOTED = re.compile(r"`([^`]+)`|'([^']+)'")
DECL = re.compile(
    r"\b(?:def|func|function)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)"
    r"|\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*="
)


def metric(message: str) -> int:
    """First integer of the message once quoted names are removed (a name like `f2` must not count)."""
    match = re.search(r"\d+", QUOTED.sub("", message))
    if not match:
        raise ToolError(f"no metric in linter message: {message!r}")
    return int(match.group())


def function_name(message: str, source_line: str) -> str:
    quoted = QUOTED.search(message)
    if quoted:
        return quoted.group(1) or quoted.group(2)
    decl = DECL.search(source_line)
    if decl:
        return decl.group(1) or decl.group(2)
    return "<anonymous>"


def make_finding(file: str, rule: str, message: str, source_line: str) -> Finding:
    if rule in FUNCTION_LIMITS:
        return Finding(file, rule, function_name(message, source_line), metric(message))
    return Finding(file, rule, "", 1)


def aggregate(findings) -> dict:
    """Function rules keep the max per (file, rule, name); count rules add up per (file, rule)."""
    out: dict = {}
    for f in findings:
        key = (f.file, f.rule, f.name)
        if f.rule in FUNCTION_LIMITS:
            out[key] = max(out.get(key, 0), f.value)
        else:
            out[key] = out.get(key, 0) + f.value
    return out


def blocking(head: dict, refs: list) -> list:
    """[(key, head_value, reference_value)] for every key above its limit that got worse than every reference."""
    out = []
    for key, value in head.items():
        reference = max([FUNCTION_LIMITS.get(key[1], 0)] + [ref.get(key, 0) for ref in refs])
        if value > reference:
            out.append((key, value, reference))
    return sorted(out)


def sccs(edges: dict) -> list:
    """Strongly connected components with more than one node (= import cycles). Iterative Tarjan."""
    state = {"index": {}, "low": {}, "stack": [], "on": set(), "out": []}
    for root in sorted(edges):
        if root not in state["index"]:
            _visit(root, edges, state)
    return state["out"]


def _visit(root, edges, state):
    index, low = state["index"], state["low"]
    _enter(root, state)
    work = [(root, iter(sorted(edges.get(root, ()))))]
    while work:
        node, children = work[-1]
        child = next(children, None)
        if child is None:
            work.pop()
            if low[node] == index[node]:
                _pop_component(node, state)
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
        elif child not in index:
            _enter(child, state)
            work.append((child, iter(sorted(edges.get(child, ())))))
        elif child in state["on"]:
            low[node] = min(low[node], index[child])


def _enter(node, state):
    state["index"][node] = state["low"][node] = len(state["index"])
    state["stack"].append(node)
    state["on"].add(node)


def _pop_component(node, state):
    component = set()
    while True:
        member = state["stack"].pop()
        state["on"].discard(member)
        component.add(member)
        if member == node:
            break
    if len(component) > 1:
        state["out"].append(frozenset(component))


def new_cycles(head_edges: dict, ref_edges: list) -> list:
    known = [cycle for edges in ref_edges for cycle in sccs(edges)]
    return sorted(sorted(c) for c in sccs(head_edges) if not any(c <= k for k in known))


HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _diff_path(header: str):
    """Path of a `+++ ` header; None for /dev/null. C-quoted paths are not decoded: fail closed."""
    if header.startswith('+++ "'):
        raise ToolError(f"quoted path in diff header (set core.quotePath=false?): {header!r}")
    return header[6:].removesuffix("\t") if header.startswith("+++ b/") else None


def added_lines(diff_text: str) -> set:
    """{(file, line, text)} added by a `git diff -U0` (no context lines)."""
    out, file, line = set(), None, 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            file = _diff_path(raw)
        elif HUNK.match(raw):
            line = int(HUNK.match(raw).group(1))
        elif raw.startswith("+") and file:
            out.add((file, line, raw[1:]))
            line += 1
    return out


def ours(base_added: set, upstream_added) -> set:
    """Fork: lines added vs the PR base that are ALSO added vs upstream (same file and text) = our lines."""
    if upstream_added is None:
        return base_added
    texts = {(file, text) for file, _, text in upstream_added}
    return {a for a in base_added if (a[0], a[2]) in texts}


# `/* eslint rule: [...] */` rewrites rule limits inline (not eslint-disable/-env/global): flagged by text, so a PR cannot
# switch the eslint-comments/no-use rule off in the same comment. Case-sensitive (ESLint directives are lowercase); also
# matches a continuation line starting with bare `eslint` but never a ` * eslint` JSDoc line.
# U+FEFF counts as whitespace: ESLint's trim() strips it and still honours the directive.
INLINE_ESLINT_CONFIG = re.compile(r"(?:/\*|^)[\s\ufeff]*eslint(?:[\s\ufeff]|$)")
SUPPRESSION = re.compile(r"(?i:noqa|nolint|eslint-disable)|" + INLINE_ESLINT_CONFIG.pattern)
NOQA = re.compile(r"#\s*noqa\b(?P<rest>.*)$", re.I)
NOQA_CODES = re.compile(r"^:\s*[A-Z]+\d+(?:[\s,]+[A-Z]+\d+)*(?P<reason>.*)$", re.I)
PY_FILE_LEVEL = re.compile(r"#\s*(?:ruff|flake8)\s*:\s*noqa", re.I)
JS_EXTENSIONS = (".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx")


def _first_code_line(line_after) -> str:
    """First line below that is not a `//` comment (doc comments may sit between //nolint and `package`).
    A blank line ends the search: golangci does not treat a //nolint separated by one as file-level."""
    for k in range(1, 101):
        below = line_after(k).strip()
        if not below.startswith("//"):
            return below
    return ""


def _suppression_problem(file: str, text: str, line_after):
    if file.endswith(".py"):
        if PY_FILE_LEVEL.search(text):
            return "file-level suppression"
        noqa = NOQA.search(text)
        codes = NOQA_CODES.match(noqa.group("rest")) if noqa else None
        if noqa and (not codes or not re.search(r"[^\W\d_]", codes.group("reason"))):
            return "noqa needs the rule code and a reason"
    elif file.endswith(".go"):
        if re.match(r"\s*//\s*nolint", text) and _first_code_line(line_after).startswith("package "):
            return "file-level suppression"
    elif file.endswith(JS_EXTENSIONS):
        if re.search(r"/\*\s*eslint-disable(?![-\w])", text):
            return "file-level suppression"
        if INLINE_ESLINT_CONFIG.search(text):
            return "inline eslint rule config"
    return None


def suppression_problems(added: set, read_line) -> list:
    """read_line(file, n) -> text of line n in the head tree (used to spot Go file-level //nolint)."""
    problems = []
    for file, line, text in sorted(added):
        problem = _suppression_problem(file, text, lambda k: read_line(file, line + k))
        if problem:
            problems.append(f"{file}:{line}: {problem}")
    return problems


def run(cmd: list, cwd, ok=(0,)) -> str:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as err:
        raise ToolError(f"cannot start {cmd[0]}: {err}") from err
    if proc.returncode not in ok:
        raise ToolError(f"{' '.join(map(str, cmd[:3]))} exited {proc.returncode}: {proc.stderr[-2000:]}")
    return proc.stdout


def line_at(tree: Path, file: str, number: int) -> str:
    try:
        lines = (tree / file).read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return ""
    return lines[number - 1].rstrip("\r") if 0 < number <= len(lines) else ""


EXTENSIONS = {"go": (".go",), "python": (".py",), "js": (".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx")}


HERE = Path(__file__).resolve().parent
# Default JS tools come from the node_modules next to this script (installed from the lockfile), never from the PR tree.
NODE_BIN = {"ESLINT": "eslint/bin/eslint.js", "DEPCRUISE": "dependency-cruiser/bin/dependency-cruiser.mjs"}


def tool(name: str, default: list) -> list:
    """Command prefix from RATCHET_<NAME> (POSIX quoting; on Windows write paths with forward slashes)."""
    env = os.environ.get(f"RATCHET_{name}")
    return shlex.split(env) if env else default


def node_tool(name: str) -> list:
    return tool(name, ["node", str(HERE / "node_modules" / NODE_BIN[name])])


def parse(raw: str, what: str):
    """json.loads that fails closed: a tool that printed nothing or garbage must not read as 'no findings'."""
    try:
        return json.loads(raw)
    except ValueError as err:
        raise ToolError(f"{what} printed no valid JSON: {err}") from err


def norm(root: str) -> str:
    return PurePosixPath(root).as_posix()


def rel(tree: Path, path: str) -> str:
    return Path(os.path.relpath(Path(path).resolve(), tree.resolve())).as_posix()


def present(tree: Path, dirs: list) -> list:
    return [norm(d) for d in dirs if (tree / d).is_dir()]


def go_findings(tree: Path, cfg: dict, cfg_dir: Path) -> list:
    out = []
    for module in present(tree, cfg["modules"]):
        raw = run(tool("GOLANGCI", ["golangci-lint"]) + ["run", "--config", str(cfg_dir / cfg["config"]),
                  "--out-format", "json", "./..."], tree / module, ok=(0, 1))
        for issue in parse(raw, "golangci-lint").get("Issues") or []:
            # golangci reports the path relative to its cwd, and spells it ../.. or absolute when the tree is reached
            # through a symlink or an 8.3 short name (Windows temp dirs): resolve it instead of concatenating.
            file = rel(tree, str(tree / module / issue["Pos"]["Filename"]))
            line = line_at(tree, file, issue["Pos"]["Line"])
            out.append(make_finding(file, issue["FromLinter"], issue["Text"], line))
    return out


def python_findings(tree: Path, cfg: dict, cfg_dir: Path) -> list:
    roots = present(tree, cfg["roots"])
    if not roots:
        return []
    raw = run(tool("RUFF", ["ruff"]) + ["check", "--config", str(cfg_dir / cfg["config"]), "--output-format", "json",
              "--exit-zero", "--no-cache", *roots], tree)
    out = []
    for diag in parse(raw, "ruff"):
        file = rel(tree, diag["filename"])
        out.append(make_finding(file, diag["code"] or "syntax-error", diag["message"],
                                line_at(tree, file, diag["location"]["row"])))
    return out


def js_findings(tree: Path, cfg: dict, cfg_dir: Path, head: bool = False) -> list:
    roots = present(tree, cfg["roots"])
    if not roots:
        return []
    raw = run(node_tool("ESLINT") + ["-c", str(cfg_dir / cfg["config"]), "-f", "json",
              "--no-error-on-unmatched-pattern", *roots], tree, ok=(0, 1))
    results = parse(raw, "eslint")
    if head and not results:
        raise ToolError(f"eslint analysed no file under {roots}: the config ignores everything, the gate would check nothing")
    out = []
    for result in results:
        file = rel(tree, result["filePath"])
        for msg in result["messages"]:
            rule = msg.get("ruleId") or "parse-error"
            out.append(make_finding(file, rule, msg["message"], line_at(tree, file, msg.get("line", 0))))
    return out


# Run with -P (Python 3.11+: no cwd on sys.path; -I would also hide user-site installs): a PR's root json.py/grimp.py
# must not be imported. The tree is appended AFTER the
# imports so the package under analysis resolves but never shadows the stdlib or grimp.
GRIMP = ("import grimp, json, os, sys; sys.path.append(os.getcwd()); g = grimp.build_graph(sys.argv[1]); "
         "print(json.dumps({m: sorted(g.find_modules_directly_imported_by(m)) for m in g.modules}))")


def python_edges(tree: Path, cfg: dict) -> dict:
    package = cfg.get("cycles_package")
    if not package or not (tree / package).is_dir():
        return {}
    raw = run(tool("PYTHON", [sys.executable]) + ["-P", "-c", GRIMP, package], tree)
    return {k: set(v) for k, v in parse(raw, "grimp").items()}


def js_edges(tree: Path, cfg: dict) -> dict:
    roots = present(tree, cfg["roots"])
    if not cfg.get("cycles") or not roots:
        return {}
    raw = run(node_tool("DEPCRUISE") + ["--no-config", "--output-type", "json", *roots], tree)
    return {
        m["source"]: {d["resolved"] for d in m["dependencies"]
                      if not d.get("coreModule") and "node_modules" not in d["resolved"]}
        for m in parse(raw, "depcruise")["modules"]
    }


RUNNERS = {"go": go_findings, "python": python_findings, "js": js_findings}
EDGES = {"python": python_edges, "js": js_edges}


def roots_of(lang_cfg) -> list:
    return [norm(r) for r in (lang_cfg or {}).get("roots") or (lang_cfg or {}).get("modules") or []]


def excluded(file: str, patterns: list) -> bool:
    return any(fnmatch.fnmatch(file, p) for p in patterns)


def in_scope(added: set, cfg: dict) -> set:
    """Added lines of files the linters actually cover (ruling R8): a configured language, under its roots, not excluded.
    Docs and test literals that merely quote `# noqa` or `eslint-disable` must not block a PR."""
    def linted(file: str) -> bool:
        return not excluded(file, cfg.get("exclude", [])) and any(
            file.endswith(EXTENSIONS[lang]) and any(PurePosixPath(file).is_relative_to(root) for root in roots_of(cfg.get(lang)))
            for lang in EXTENSIONS)
    return {a for a in added if linted(a[0])}


def check_roots(tree: Path, cfg: dict) -> None:
    for lang, exts in EXTENSIONS.items():
        for root in roots_of(cfg.get(lang)):
            files = (p for p in (tree / root).rglob("*") if p.suffix in exts and "node_modules" not in p.parts)
            if next(files, None) is None:
                raise ToolError(f"{lang} root {root!r} has no {'/'.join(exts)} files: the gate would check nothing")


def analyse(tree: Path, cfg: dict, cfg_dir: Path, head: bool = False):
    runners = {**RUNNERS, "js": partial(js_findings, head=head)}
    findings = [f for lang, fn in runners.items() if lang in cfg for f in fn(tree, cfg[lang], cfg_dir)]
    kept = [f for f in findings if not excluded(f.file, cfg.get("exclude", []))]
    edges = {lang: fn(tree, cfg[lang]) for lang, fn in EDGES.items() if lang in cfg}
    return aggregate(kept), edges


# --text/--no-textconv: a PR's .gitattributes (`*.py -diff`) must not turn the diff into "Binary files differ".
DIFF = ["-c", "core.quotePath=false", "diff", "-U0", "--no-color", "--no-ext-diff", "--text", "--no-textconv",
        "--src-prefix=a/", "--dst-prefix=b/"]


def git(*args: str) -> str:
    return run(["git", *args], ".")


def with_trees(refs: list, fn) -> list:
    with tempfile.TemporaryDirectory() as tmp:
        trees = []
        try:
            for i, ref in enumerate(refs):
                path = Path(tmp) / f"ref{i}"
                git("worktree", "add", "--detach", "--force", str(path), ref)
                trees.append(path)
            return [fn(t) for t in trees]
        finally:
            for t in trees:
                subprocess.run(["git", "worktree", "remove", "--force", str(t)], capture_output=True)


def evaluate(args) -> dict:
    cfg_path = Path(args.config).resolve()
    cfg, head = json.loads(cfg_path.read_text(encoding="utf-8")), Path.cwd()
    check_roots(head, cfg)
    refs = [args.base] + ([git("merge-base", args.upstream, "HEAD").strip()] if args.upstream else [])
    head_values, head_edges = analyse(head, cfg, cfg_path.parent, head=True)
    ref_results = with_trees(refs, lambda tree: analyse(tree, cfg, cfg_path.parent))
    added = in_scope(ours(added_lines(git(*DIFF, f"{args.base}...HEAD")),
                          added_lines(git(*DIFF, f"{args.upstream}...HEAD")) if args.upstream else None), cfg)
    blocked = blocking(head_values, [values for values, _ in ref_results])
    blocked_keys = {key for key, _, _ in blocked}
    return {
        "blocking": blocked,
        "cycles": [c for lang in head_edges for c in new_cycles(head_edges[lang], [e[lang] for _, e in ref_results])],
        "suppressions": suppression_problems(added, lambda f, n: line_at(head, f, n)),
        "added_suppressions": sorted(a for a in added if SUPPRESSION.search(a[2])),
        "debt": sorted((k, v) for k, v in head_values.items() if k not in blocked_keys),
    }


def _section(title: str, items: list) -> list:
    return [f"### {title} ({len(items)})", *[f"- {i}" for i in items], ""]


def write_summary(report: dict) -> None:
    lines = ["## principles-mechanical", ""]
    lines += _section("Blocking: above the limit and worse than the reference",
                      [f"`{f}` {rule} {name}: {v} (reference {ref})" for (f, rule, name), v, ref in report["blocking"]])
    lines += _section("New import cycles", [" → ".join(c) for c in report["cycles"]])
    lines += _section("Suppression problems", report["suppressions"])
    lines += _section("Suppressions added by this PR", [f"`{f}:{n}` {t.strip()}" for f, n, t in report["added_suppressions"]])
    lines += _section("Debt: above the limit, not worse (does not block)",
                      [f"`{f}` {rule} {name}: {v}" for (f, rule, name), v in report["debt"]])
    text = "\n".join(lines) + "\n"
    print(text)
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Principles ratchet (review system v3).")
    parser.add_argument("--config", required=True, help="repo.json, read from the PR base")
    parser.add_argument("--base", required=True, help="PR base ref, e.g. origin/main")
    parser.add_argument("--upstream", help="fork only: upstream ref, e.g. refs/xpand/upstream-main")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args)
    except (ToolError, OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        print(f"principles: TOOL ERROR, the gate fails closed: {type(err).__name__}: {err}", file=sys.stderr)
        return 2
    write_summary(report)
    return 1 if report["blocking"] or report["cycles"] or report["suppressions"] else 0


if __name__ == "__main__":
    sys.exit(main())
