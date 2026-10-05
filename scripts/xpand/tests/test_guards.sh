#!/usr/bin/env bash
# scripts/xpand/tests/test_guards.sh, self-test of the XPAND guards (VOZ-AT-B9-02, VOZ-AT-B9-05)
# Runs each guard against a throwaway repo: positive controls must pass, negative controls must fail
# with the full error shape (code, where, reason, hint; VOZ-AC-B0-28).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

# The forbidden words are assembled at runtime so this fixture file itself carries no attribution text.
ai="Cla""ude"; vendor="Anthr""opic"

expect_fail() { # <label> <cmd...>
  local label="$1"; shift
  local err
  if err="$("$@" 2>&1 >/dev/null)"; then echo "FAIL: $label not detected"; exit 1; fi
  for key in code= where= reason= hint=; do
    [[ "$err" == *"$key"* ]] || { echo "FAIL: $label: error lacks '$key': $err"; exit 1; }
  done
  echo "PASS $label"
}

cd "$tmp" && git init -q -b main && git config user.email t@t && git config user.name t
echo a > a.txt && git add a.txt && git commit -qm "base" && git branch base

# --- check_no_ai_attribution.sh <base-ref>
git checkout -q -b fix/attr
echo b >> a.txt && git commit -qam "fix: clean change"
"$here/check_no_ai_attribution.sh" base && echo "PASS clean range"
git commit -q --allow-empty -m "fix: x" -m "Co-Authored-By: $ai <noreply@$(echo "$vendor" | tr A-Z a-z).com>"
expect_fail "trailer detected" "$here/check_no_ai_attribution.sh" base
git reset -q --hard HEAD~1
git commit -q --allow-empty -m "fix: y mentions $(echo "$vendor" | tr a-z A-Z) in the subject"
expect_fail "case-insensitive mention detected" "$here/check_no_ai_attribution.sh" base
expect_fail "unknown base ref fails closed" "$here/check_no_ai_attribution.sh" no-such-ref
git -c user.name="$ai" commit -q --allow-empty -m "fix: z clean message"
expect_fail "author name detected" "$here/check_no_ai_attribution.sh" base
git reset -q --hard HEAD~1
GIT_COMMITTER_NAME="$ai" git commit -q --allow-empty -m "fix: w clean message"
expect_fail "committer name detected" "$here/check_no_ai_attribution.sh" base
git reset -q --hard HEAD~1
printf 'fix/ok\nfix: clean title\n' | "$here/check_no_ai_attribution.sh" --stdin && echo "PASS clean title and branch"
expect_fail "branch name detected via stdin" bash -c "echo fix/$ai-thing | '$here/check_no_ai_attribution.sh' --stdin"
git checkout -q main

# --- check_fix_against_g0.sh <upstream> [<fork-base>]
git checkout -q -b fix/clean base
echo c > c.txt && git add c.txt && git commit -qm "fix: new file"
"$here/check_fix_against_g0.sh" HEAD base && echo "PASS clean fix, upstream = HEAD"

git checkout -q -b up-other base
echo u > u.txt && git add u.txt && git commit -qm "upstream: other file"
"$here/check_fix_against_g0.sh" up-other base && echo "PASS clean fix vs non-conflicting upstream"

git checkout -q -b up-conflict base
echo upstream-side > a.txt && git commit -qam "upstream: edits a.txt"
git checkout -q -b fix/conflict base
echo fork-side > a.txt && git commit -qam "fix: edits a.txt"
expect_fail "new conflict outside the G0 list detected" "$here/check_fix_against_g0.sh" up-conflict base
expect_fail "unknown upstream ref fails closed" "$here/check_fix_against_g0.sh" no-such-ref base

git checkout -q -b fix/touches-g0 base
mkdir -p api/routes && echo x > api/routes/user.py && git add . && git commit -qm "fix: touches a G0 conflict file"
expect_fail "touching api/routes/user.py detected" "$here/check_fix_against_g0.sh" HEAD base

# Upstream conflicts ONLY on an allowed (G0 list) file that the fork base already diverged on; the fix touches
# nothing in the list -> must pass (guards the allowed-list filter and the merge-tree informational lines).
git checkout -q -b g0-base base
mkdir -p api/routes && echo base > api/routes/user.py && git add api && git commit -qm "base: user.py"
git checkout -q -b up-allowed g0-base
echo upstream-side > api/routes/user.py && git commit -qam "upstream: edits user.py"
git checkout -q -b fork-base g0-base
echo fork-side > api/routes/user.py && git commit -qam "fork base: edits user.py"
git checkout -q -b fix/allowed-conflict-only fork-base
echo d > d.txt && git add d.txt && git commit -qm "fix: unrelated file"
"$here/check_fix_against_g0.sh" up-allowed fork-base && echo "PASS conflict only on an allowed file, clean fix"
