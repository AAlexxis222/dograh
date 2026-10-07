"""python -m unittest discover -s tools/principles -p 'test_*.py'"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ratchet as r  # noqa: E402 import after sys.path insert

F = r.Finding


class Metric(unittest.TestCase):
    def test_messages_of_each_linter(self):
        self.assertEqual(r.metric("`f2` is too complex (16 > 15)"), 16)                      # ruff C901
        self.assertEqual(r.metric("Too many statements (58 > 50)"), 58)                      # ruff PLR0915
        self.assertEqual(r.metric("cyclomatic complexity 16 of func `run2` is high (> 15)"), 16)  # gocyclo
        self.assertEqual(r.metric("Function 'big' is too long (90 > 80)"), 90)               # funlen
        self.assertEqual(r.metric("Function 'f' has a complexity of 16. Maximum allowed is 15."), 16)
        self.assertEqual(r.metric("Arrow function has too many lines (91). Maximum allowed is 80."), 91)

    def test_no_number_fails_closed(self):
        with self.assertRaises(r.ToolError):
            r.metric("weird message")


class Names(unittest.TestCase):
    def test_quoted_name_wins(self):
        self.assertEqual(r.function_name("`(*S).Run` is high", "func (s *S) Run() {"), "(*S).Run")

    def test_from_source_line(self):
        self.assertEqual(r.function_name("Too many statements (58 > 50)", "    def create_stt_service(cfg):"), "create_stt_service")
        self.assertEqual(r.function_name("x", "func (s *S) Run(ctx context.Context) error {"), "Run")
        self.assertEqual(r.function_name("Arrow function has a complexity of 16.", "export const load = async () => {"), "load")
        self.assertEqual(r.function_name("Arrow function has a complexity of 16.", "  items.map((x) => {"), "<anonymous>")


class Blocking(unittest.TestCase):
    key = ("a.py", "C901", "f")

    def check(self, head, refs):
        return [b[0] for b in r.blocking({self.key: head}, [{self.key: v} if v else {} for v in refs])]

    def test_new_function_above_limit_blocks(self):
        self.assertEqual(self.check(16, [None]), [self.key])

    def test_existing_function_crossing_the_limit_blocks(self):
        self.assertEqual(self.check(16, [None]), [self.key])  # 15 at base = not reported = limit

    def test_unchanged_debt_does_not_block(self):
        self.assertEqual(self.check(30, [30]), [])

    def test_worse_debt_blocks(self):
        self.assertEqual(self.check(31, [30]), [self.key])

    def test_upstream_growth_is_not_ours(self):
        self.assertEqual(self.check(40, [30, 40]), [])
        self.assertEqual(self.check(41, [30, 40]), [self.key])

    def test_count_rules_ratchet_per_file(self):
        k = ("a.py", "S110", "")
        self.assertEqual(r.blocking({k: 3}, [{k: 2}]), [(k, 3, 2)])
        self.assertEqual(r.blocking({k: 2}, [{k: 2}]), [])
        self.assertEqual(r.blocking({k: 1}, [{}]), [(k, 1, 0)])

    def test_homonyms_compare_by_max(self):
        agg = r.aggregate([F("a.py", "C901", "__init__", 20), F("a.py", "C901", "__init__", 17)])
        self.assertEqual(agg, {("a.py", "C901", "__init__"): 20})

    def test_no_findings_passes(self):
        self.assertEqual(r.blocking({}, [{}]), [])


class Cycles(unittest.TestCase):
    def test_finds_two_and_three_cycles(self):
        edges = {"a": {"b"}, "b": {"a"}, "c": {"d"}, "d": {"e"}, "e": {"c"}, "x": {"y"}}
        self.assertEqual(sorted(sorted(c) for c in r.sccs(edges)), [["a", "b"], ["c", "d", "e"]])

    def test_only_new_cycles_count(self):
        ref = {"a": {"b"}, "b": {"a"}}
        self.assertEqual(r.new_cycles({"a": {"b"}, "b": {"a"}}, [ref]), [])
        self.assertEqual(r.new_cycles({"a": {"b"}, "b": {"a"}, "p": {"q"}, "q": {"p"}}, [ref]), [["p", "q"]])


DIFF = """diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -3,0 +4,2 @@ def f():
+    try: g()
+    except Exception: pass  # noqa: S110
diff --git a/gone.py b/gone.py
--- a/gone.py
+++ /dev/null
@@ -1 +0,0 @@
-x = 1
"""


class Diffs(unittest.TestCase):
    def test_added_lines(self):
        self.assertEqual(r.added_lines(DIFF), {("x.py", 4, "    try: g()"), ("x.py", 5, "    except Exception: pass  # noqa: S110")})

    def test_path_with_space_and_unsupported_headers(self):
        d = "\n".join(["--- a/my x.py\t", "+++ b/my x.py\t", "@@ -1,0 +1 @@", "+y = 1"])
        self.assertEqual(r.added_lines(d), {("my x.py", 1, "y = 1")})
        quoted = "\n".join(['--- "a/\\303\\261.py"', '+++ "b/\\303\\261.py"', "@@ -1,0 +1 @@", "+y = 1"])
        with self.assertRaises(r.ToolError):
            r.added_lines(quoted)

    def test_ours_is_the_intersection_by_text(self):
        base = {("x.py", 4, "a"), ("x.py", 9, "upstream line")}
        up = {("x.py", 40, "a")}
        self.assertEqual(r.ours(base, up), {("x.py", 4, "a")})
        self.assertEqual(r.ours(base, None), base)


class Suppressions(unittest.TestCase):
    def problems(self, file, text, next_line=""):
        return r.suppression_problems({(file, 1, text)}, lambda f, n: next_line)

    def test_python(self):
        self.assertEqual(self.problems("a.py", "x()  # noqa: S110 retried by the caller"), [])
        self.assertTrue(self.problems("a.py", "x()  # noqa: S110"))
        self.assertTrue(self.problems("a.py", "x()  # noqa"))
        self.assertTrue(self.problems("a.py", "# ruff: noqa: C901"))

    def test_flake8_file_level_and_code_lists(self):
        self.assertTrue(self.problems("a.py", "# flake8: noqa"))
        self.assertTrue(self.problems("a.py", "# flake8: noqa: E501"))
        self.assertTrue(self.problems("a.py", "x()  # noqa: S110 BLE001"))
        self.assertTrue(self.problems("a.py", "x()  # noqa: S110, BLE001"))
        self.assertTrue(self.problems("a.py", "x()  # noqa: S110 -"))
        self.assertEqual(self.problems("a.py", "x()  # noqa:S110 retried by caller"), [])
        self.assertEqual(self.problems("a.py", "x()  # noqa: S110, BLE001 retried by caller"), [])

    def test_only_code_files_are_checked(self):
        for name in ("README.md", "ci.yml", "notes.txt"):
            self.assertEqual(self.problems(name, "/* eslint-disable complexity */"), [])
            self.assertEqual(self.problems(name, "x  # noqa"), [])
        for name in ("a.js", "a.mjs", "a.cjs", "a.jsx", "a.ts", "a.tsx"):
            self.assertTrue(self.problems(name, "/* eslint-disable complexity */"))

    def test_js(self):
        self.assertTrue(self.problems("a.ts", "/* eslint-disable complexity -- legacy */"))
        self.assertEqual(self.problems("a.ts", "// eslint-disable-next-line complexity -- legacy"), [])

    def test_js_inline_rule_config(self):
        off = '/* eslint @eslint-community/eslint-comments/no-use: "off", complexity: ["error", 100] -- x */'
        for text in (off, "/*eslint complexity: 100*/", "/* eslint complexity: 0 */", "/*	eslint x: 0*/",
                     "  eslint complexity: [2, 100] */", "eslint", "/*\ufeffeslint complexity: 100 */"):
            self.assertEqual(len(self.problems("a.ts", text)), 1, text)
            self.assertTrue(r.SUPPRESSION.search(text), text)  # listed under "Suppressions added by this PR"
        for text in ("/* eslint-env node */", "/* global foo */", "/* exported bar */", "const eslintConfig = 1",
                     "// eslint-disable-next-line complexity -- legacy", "/* eslint-enable */",
                     " * Eslint runs in CI", " * eslint is configured", "/* Eslint complexity: 5 */"):
            self.assertEqual(self.problems("a.ts", text), [], text)

    def test_main_exits_2_when_json_has_the_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp, "repo.json")
            cfg.write_text("[]")  # an array where an object is expected -> AttributeError
            self.assertEqual(r.main(["--config", str(cfg), "--base", "x"]), 2)

    def test_the_gate_files_pass_their_own_suppression_check(self):
        here = Path(r.__file__).parent
        for path in here.glob("*.mjs"):  # the JS the plugin's repo.json lints under tools/
            lines = path.read_text(encoding="utf-8").split("\n")
            added = {(path.name, n, text) for n, text in enumerate(lines, 1)}
            read = lambda f, n, lines=lines: lines[n - 1] if 0 < n <= len(lines) else ""
            self.assertEqual(r.suppression_problems(added, read), [], path.name)

    def test_go_nolint_above_doc_comments_is_file_level(self):
        source = ["//nolint:errcheck // legacy file", "// Package x does y.", "//", "// More docs.", "package x"]
        read = lambda f, n: source[n - 1] if 0 < n <= len(source) else ""
        self.assertTrue(r.suppression_problems({("a.go", 1, source[0])}, read))
        func = ["//nolint:errcheck // legacy", "// run does z.", "func run() {"]
        read = lambda f, n: func[n - 1] if 0 < n <= len(func) else ""
        self.assertEqual(r.suppression_problems({("a.go", 1, func[0])}, read), [])
        blank = ["//nolint:errcheck // legacy", "", "package x"]  # golangci does not treat this one as file-level
        read = lambda f, n: blank[n - 1] if 0 < n <= len(blank) else ""
        self.assertEqual(r.suppression_problems({("a.go", 1, blank[0])}, read), [])

    def test_go_file_level(self):
        self.assertTrue(self.problems("a.go", "//nolint:gocyclo // legacy", "package main"))
        self.assertEqual(self.problems("a.go", "//nolint:gocyclo // legacy", "func big() {"), [])


class Tools(unittest.TestCase):
    def test_unexpected_exit_code_fails_closed(self):
        with self.assertRaises(r.ToolError):
            r.run([sys.executable, "-c", "import sys; sys.exit(3)"], ".", ok=(0, 1))

    def test_missing_tool_fails_closed(self):
        with self.assertRaises(r.ToolError):
            r.run(["definitely-not-a-tool-xyz"], ".")

    def test_line_at_survives_bad_bytes_and_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.py").write_bytes(b"x = 1\r\ndef f(\xff):\r\n")
            self.assertEqual(r.line_at(Path(tmp), "a.py", 2).startswith("def f("), True)
            self.assertEqual(r.line_at(Path(tmp), "a.py", 99), "")
            Path(tmp, "b.py").write_bytes(b"a\x0cb\nsecond\n")  # form feed must not split lines
            self.assertEqual(r.line_at(Path(tmp), "b.py", 2), "second")
            self.assertEqual(r.line_at(Path(tmp), "missing.py", 1), "")


DATA = Path(__file__).parent / "testdata"


def fake_tool(fixture: Path, tree: Path) -> str:
    """A command that prints the fixture with {TREE} replaced (absolute paths for ruff/eslint)."""
    slash = lambda p: str(p).replace(chr(92), "/")  # POSIX shlex eats backslashes
    code = f"print(open('{slash(fixture)}').read().replace('{{TREE}}', '{slash(tree)}'))"
    return f'"{slash(sys.executable)}" -c "{code}"'


class Runners(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Path(self.tmp.name)
        for rel in ("cli/m/a.go", "api/a.py", "ui/a.ts"):
            (self.tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.tree / rel).write_text("func big() {\ndef big():\nfunction big() {\n")

    def tearDown(self):
        self.tmp.cleanup()

    def with_env(self, name, fixture):
        os.environ[f"RATCHET_{name}"] = fake_tool(DATA / fixture, self.tree)
        self.addCleanup(os.environ.pop, f"RATCHET_{name}")

    def test_golangci(self):
        self.with_env("GOLANGCI", "golangci.json")
        found = r.go_findings(self.tree, {"modules": ["cli/m"], "config": "golangci.yml"}, self.tree)
        self.assertTrue(any(f.rule == "gocyclo" and f.file.startswith("cli/m/") and f.value > 15 for f in found))

    def test_golangci_path_spelled_absolute_or_with_dotdot(self):
        self.addCleanup(os.environ.pop, "RATCHET_GOLANGCI", None)
        issue = {"FromLinter": "gocyclo", "Text": "cyclomatic complexity 17 of func `big` is high (> 15)"}
        for spelling in ("{TREE}/cli/m/a.go", "../../x/../cli/m/a.go"):
            fixture = self.tree / "g.json"
            fixture.write_text(json.dumps({"Issues": [{**issue, "Pos": {"Filename": spelling, "Line": 1}}]}))
            os.environ["RATCHET_GOLANGCI"] = fake_tool(fixture, self.tree)
            found = r.go_findings(self.tree, {"modules": ["cli/m"], "config": "golangci.yml"}, self.tree)
            self.assertEqual([f.file for f in found], ["cli/m/a.go"], spelling)

    def test_ruff(self):
        self.with_env("RUFF", "ruff.json")
        found = r.python_findings(self.tree, {"roots": ["api"], "config": "ruff.toml"}, self.tree)
        self.assertTrue(any(f.rule == "C901" and f.file == "api/a.py" for f in found))

    def test_eslint(self):
        self.with_env("ESLINT", "eslint.json")
        found = r.js_findings(self.tree, {"roots": ["ui"], "config": "eslint.config.mjs"}, self.tree)
        self.assertTrue(any(f.rule == "complexity" and f.file == "ui/a.ts" for f in found))

    def test_absent_root_in_reference_is_skipped(self):
        self.assertEqual(r.python_findings(self.tree, {"roots": ["nope"], "config": "ruff.toml"}, self.tree), [])

    def test_empty_root_at_head_fails_closed(self):
        with self.assertRaises(r.ToolError):
            r.check_roots(self.tree, {"python": {"roots": ["ui"]}})


class Configs(unittest.TestCase):
    def test_configs_match_limits(self):
        here = Path(r.__file__).parent
        golangci = (here / "golangci.yml").read_text()
        ruff = (here / "ruff.toml").read_text()
        eslint = (here / "eslint.config.mjs").read_text()
        self.assertIn(f"min-complexity: {r.FUNCTION_LIMITS['gocyclo']}", golangci)
        self.assertIn(f"lines: {r.FUNCTION_LIMITS['funlen']}", golangci)
        self.assertIn("statements: -1", golangci)
        self.assertIn(f"max-complexity = {r.FUNCTION_LIMITS['C901']}", ruff)
        self.assertIn(f"max-statements = {r.FUNCTION_LIMITS['PLR0915']}", ruff)
        self.assertIn(f"complexity: ['error', {r.FUNCTION_LIMITS['complexity']}]", eslint)
        self.assertIn(f"max: {r.FUNCTION_LIMITS['max-lines-per-function']}", eslint)
        self.assertNotIn("BLE001", ruff)
        self.assertIn("respect-gitignore = false", ruff)
        self.assertIn("exclude = []", ruff)

    def test_ruff_sees_files_a_pr_gitignores_or_puts_in_default_excluded_dirs(self):
        probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True)
        if probe.returncode:
            self.skipTest("ruff not installed")
        body = "def big(x):\n    y = 0\n" + "".join(f"    if x == {i}:\n        y += 1\n" for i in range(16)) + "    return y\n"
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            for rel in ("api/x/c.py", "api/venv/c.py"):
                (tree / rel).parent.mkdir(parents=True)
                (tree / rel).write_text(body)
            (tree / "api/x/.gitignore").write_text("c.py\n")
            os.environ.pop("RATCHET_RUFF", None)
            os.environ["RATCHET_RUFF"] = f'"{sys.executable}" -m ruff'.replace(chr(92), "/")
            self.addCleanup(os.environ.pop, "RATCHET_RUFF", None)
            here = Path(r.__file__).parent
            found = r.python_findings(tree, {"roots": ["api"], "config": "ruff.toml"}, here)
        self.assertEqual(sorted(f.file for f in found if f.rule == "C901"), ["api/venv/c.py", "api/x/c.py"])


class InScope(unittest.TestCase):
    def test_only_linted_files_keep_their_suppression_checks(self):
        cfg = {"exclude": ["*/tests/*", "*_test.go"], "python": {"roots": ["api"]}, "go": {"modules": ["cli/m"]}}
        line = lambda f: (f, 1, "x = 1  # noqa")
        files = ["docs/guide.md", "api/tests/t.py", "cli/m/a_test.go", "other/a.py", "api2/a.py", "api/a.py", "cli/m/a.go"]
        kept = r.in_scope({line(f) for f in files}, cfg)
        self.assertEqual(sorted(f for f, _, _ in kept), ["api/a.py", "cli/m/a.go"])


class FailClosed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Path(self.tmp.name)
        for rel in ("cli/m/a.go", "api/a.py", "ui/a.ts"):
            (self.tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.tree / rel).write_text("x\n")
        self.addCleanup(self.tmp.cleanup)

    def printing(self, name, content):
        fixture = self.tree / "out.txt"
        fixture.write_text(content)
        silent = f'"{sys.executable}" -c "pass"'.replace(chr(92), "/")  # prints nothing at all
        os.environ[f"RATCHET_{name}"] = fake_tool(fixture, self.tree) if content else silent
        self.addCleanup(os.environ.pop, f"RATCHET_{name}", None)

    def test_unreadable_json_is_a_tool_error_for_every_runner(self):
        runners = {
            "GOLANGCI": lambda: r.go_findings(self.tree, {"modules": ["cli/m"], "config": "c"}, self.tree),
            "RUFF": lambda: r.python_findings(self.tree, {"roots": ["api"], "config": "c"}, self.tree),
            "ESLINT": lambda: r.js_findings(self.tree, {"roots": ["ui"], "config": "c"}, self.tree),
            "DEPCRUISE": lambda: r.js_edges(self.tree, {"roots": ["ui"], "cycles": True}),
        }
        for name, call in runners.items():
            for content in ("", "{"):
                self.printing(name, content)
                with self.assertRaises(r.ToolError, msg=f"{name} printing {content!r}"):
                    call()

    def test_main_exits_2_on_unreadable_config(self):
        self.assertEqual(r.main(["--config", str(self.tree / "missing.json"), "--base", "x"]), 2)

    def test_head_tree_where_eslint_analysed_nothing_fails_closed(self):
        self.printing("ESLINT", "[]")
        cfg = {"roots": ["ui"], "config": "c"}
        self.assertEqual(r.js_findings(self.tree, cfg, self.tree), [])  # a reference tree may be empty
        with self.assertRaises(r.ToolError):
            r.js_findings(self.tree, cfg, self.tree, head=True)


class Roots(unittest.TestCase):
    def test_trailing_slash_and_dot(self):
        added = {("ui/a.ts", 1, "x"), ("api/b.py", 1, "x"), ("README.md", 1, "x")}
        self.assertEqual({f for f, _, _ in r.in_scope(added, {"js": {"roots": ["ui/"]}})}, {"ui/a.ts"})
        self.assertEqual({f for f, _, _ in r.in_scope(added, {"js": {"roots": ["."]}, "python": {"roots": ["."]}})},
                         {"ui/a.ts", "api/b.py"})

    def test_present_and_check_roots_normalise(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            (tree / "ui").mkdir()
            (tree / "ui" / "a.ts").write_text("x")
            self.assertEqual(r.present(tree, ["ui/", "."]), ["ui", "."])
            r.check_roots(tree, {"js": {"roots": ["ui/"]}})
            r.check_roots(tree, {"js": {"roots": ["."]}})


class Hardening(unittest.TestCase):
    def test_default_js_tools_live_next_to_the_script_not_in_the_pr_tree(self):
        for name in r.NODE_BIN:
            os.environ.pop(f"RATCHET_{name}", None)
            self.assertEqual(Path(r.node_tool(name)[1]), Path(r.__file__).resolve().parent / "node_modules" / r.NODE_BIN[name])
        os.environ["RATCHET_ESLINT"] = "custom-eslint --x"
        self.addCleanup(os.environ.pop, "RATCHET_ESLINT")
        self.assertEqual(r.node_tool("ESLINT"), ["custom-eslint", "--x"])

    def test_diff_survives_a_prs_gitattributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = lambda *a: r.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], tmp)
            git("init", "-q", "-b", "main")
            Path(tmp, "a.py").write_text("x = 1\n")
            git("add", ".")
            git("commit", "-q", "-m", "base")
            git("checkout", "-q", "-b", "pr")
            Path(tmp, ".gitattributes").write_text("*.py -diff\n")
            Path(tmp, "a.py").write_text("x = 1\ny = 2  # noqa\n")
            git("add", ".")
            git("commit", "-q", "-m", "pr")
            added = r.added_lines(r.run(["git", *r.DIFF, "main...HEAD"], tmp))
            self.assertIn(("a.py", 2, "y = 2  # noqa"), added)

    def test_grimp_does_not_import_from_the_pr_tree(self):
        try:
            import grimp  # noqa: F401 grimp is only needed by repos with a python cycles_package
        except ImportError:
            self.skipTest("grimp not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            (tree / "pkg").mkdir()
            (tree / "pkg" / "__init__.py").write_text("")
            (tree / "pkg" / "a.py").write_text("from pkg import b\n")
            (tree / "pkg" / "b.py").write_text("")
            (tree / "json.py").write_text("print('PR CODE RAN')\n")
            os.environ.pop("RATCHET_PYTHON", None)
            edges = r.python_edges(tree, {"cycles_package": "pkg"})
        self.assertIn("pkg.b", edges["pkg.a"])


if __name__ == "__main__":
    unittest.main()
