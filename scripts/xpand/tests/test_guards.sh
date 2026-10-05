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

expect_pass() { # <label> <cmd...>
  local label="$1"; shift
  "$@" >/dev/null 2>&1 || { echo "FAIL: $label"; exit 1; }
  echo "PASS $label"
}

expect_eq() { # <label> <actual> <expected>
  [[ "$2" == "$3" ]] || { echo "FAIL: $1: got '$2', want '$3'"; exit 1; }
  echo "PASS $1"
}

cd "$tmp" && git init -q -b main && git config user.email t@t && git config user.name t
echo a > a.txt && git add a.txt && git commit -qm "base" && git branch base

# --- check_no_ai_attribution.sh <base-ref>
git checkout -q -b fix/attr
echo b >> a.txt && git commit -qam "fix: clean change"
expect_pass "clean range" "$here/check_no_ai_attribution.sh" base
git commit -q --allow-empty -m "fix: x" -m "Co-Authored-By: $ai <noreply@$(echo "$vendor" | tr A-Z a-z).com>"
expect_fail "trailer detected" "$here/check_no_ai_attribution.sh" base
git reset -q --hard HEAD~1
git commit -q --allow-empty -m "fix: y mentions $(echo "$vendor" | tr a-z A-Z) in the subject"
expect_fail "case-insensitive mention detected" "$here/check_no_ai_attribution.sh" base
git reset -q --hard HEAD~1
expect_fail "unknown base ref fails closed" "$here/check_no_ai_attribution.sh" no-such-ref
# author only (committer stays clean), so the %an%n%ae part of the format is what is being proven
GIT_AUTHOR_NAME="$ai" GIT_AUTHOR_EMAIL="a@a.test" git commit -q --allow-empty -m "fix: z clean message"
expect_fail "author name detected" "$here/check_no_ai_attribution.sh" base
git reset -q --hard HEAD~1
GIT_COMMITTER_NAME="$ai" git commit -q --allow-empty -m "fix: w clean message"
expect_fail "committer name detected" "$here/check_no_ai_attribution.sh" base
git reset -q --hard HEAD~1
expect_pass "clean title and branch" bash -c "printf 'fix/ok\nfix: clean title\n' | '$here/check_no_ai_attribution.sh' --stdin"
expect_fail "branch name detected via stdin" bash -c "echo fix/$ai-thing | '$here/check_no_ai_attribution.sh' --stdin"
# head-ref argument (base-run workflow scans the fetched PR head, never HEAD): both directions must bite
git checkout -q -b fix/dirty-ref
git commit -q --allow-empty -m "fix: v mentions $ai"
git checkout -q fix/attr   # HEAD clean, dirty ref given
expect_fail "head-ref argument: dirty ref fails while HEAD is clean" "$here/check_no_ai_attribution.sh" base fix/dirty-ref
git checkout -q fix/dirty-ref   # HEAD dirty, clean ref given
expect_pass "head-ref argument: clean ref passes while HEAD is dirty" "$here/check_no_ai_attribution.sh" base fix/attr
# exclude argument: an upstream-style commit with a trailer inside the range, plus own clean commits
git checkout -q -b up-trailer base
git commit -q --allow-empty -m "upstream: x" -m "Co-Authored-By: $ai <noreply@$(echo "$vendor" | tr A-Z a-z).com>"
git checkout -q -b fix/own-on-upstream
git commit -q --allow-empty -m "fix: own clean commit"
expect_fail "upstream trailer in range fails when not excluded" "$here/check_no_ai_attribution.sh" base fix/own-on-upstream
expect_pass "upstream trailer in range passes when upstream ref excluded" "$here/check_no_ai_attribution.sh" base fix/own-on-upstream up-trailer
git commit -q --allow-empty -m "fix: own dirty $ai"
expect_fail "own dirty commit still fails with upstream excluded" "$here/check_no_ai_attribution.sh" base fix/own-on-upstream up-trailer
git checkout -q main

# --- check_fix_against_g0.sh <upstream> [<fork-base>]
git checkout -q -b fix/clean base
echo c > c.txt && git add c.txt && git commit -qm "fix: new file"
expect_pass "clean fix, upstream = HEAD" "$here/check_fix_against_g0.sh" HEAD base

git checkout -q -b up-other base
echo u > u.txt && git add u.txt && git commit -qm "upstream: other file"
expect_pass "clean fix vs non-conflicting upstream" "$here/check_fix_against_g0.sh" up-other base

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
expect_pass "conflict only on an allowed file, clean fix" "$here/check_fix_against_g0.sh" up-allowed fork-base

# --- pr-gate/reviewed-sha.sh <body> <head> <base-ref>
# Each case builds its own repo: main (a.txt) -> pr branch with the reviewed commit r1.
rs="$here/pr-gate/reviewed-sha.sh"
mkpr() { # <dir>  -> repo with branch pr at r1, body.md naming r1 (visible Review section)
  rm -rf "$1" && mkdir "$1" && cd "$1" && git init -q -b main && git config user.email t@t && git config user.name t
  echo a > a.txt && git add a.txt && git commit -qm base
  git checkout -q -b pr && echo r > r.txt && git add r.txt && git commit -qm "feat: reviewed"
  r1="$(git rev-parse HEAD)"
  printf '## Review
VERDICT: SAFE TO MERGE
REVIEWED: %s
' "$r1" > body.md
  git checkout -q main && echo m > m.txt && git add m.txt && git commit -qm "main: advances"
  git checkout -q pr
}
sha_for() { bash "$rs" body.md "$(git rev-parse HEAD)" main; } # -> prints the sha the body must match

mkpr "$tmp/rs1"
expect_eq "reviewed-sha: REVIEWED = head accepted" "$(sha_for)" "$r1"

mkpr "$tmp/rs2"; git merge -q --no-edit main
expect_eq "reviewed-sha: clean merge of base after review accepted" "$(sha_for)" "$r1"

mkpr "$tmp/rs3"
# old REVIEWED line inside an HTML comment INSIDE the Review section: only visible() keeps it from winning
printf '## Review
<!--
REVIEWED: %s
-->
VERDICT: SAFE TO MERGE
REVIEWED: %s
' "$(git rev-parse main)" "$r1" > body.md
git merge -q --no-edit main
expect_eq "reviewed-sha: REVIEWED inside an HTML comment ignored" "$(sha_for)" "$r1"

mkpr "$tmp/rs4"; git merge -q --no-commit --no-ff main && echo evil > evil.txt && git add evil.txt && git commit -qm "Merge main (with extra edit)"
expect_eq "reviewed-sha: evil merge (extra edit) forces head" "$(sha_for)" "$(git rev-parse HEAD)"

mkpr "$tmp/rs5"; git checkout -q -b other main && echo o > o.txt && git add o.txt && git commit -qm "other branch" && git checkout -q pr && git merge -q --no-edit other
expect_eq "reviewed-sha: merge of a non-base branch forces head" "$(sha_for)" "$(git rev-parse HEAD)"

mkpr "$tmp/rs6"; echo more > more.txt && git add more.txt && git commit -qm "feat: own commit after review"
expect_eq "reviewed-sha: own commit after review forces head" "$(sha_for)" "$(git rev-parse HEAD)"

mkpr "$tmp/rs7"
git checkout -q main && echo main-side > r.txt && git add r.txt && git commit -qm "main: edits r.txt" && git checkout -q pr
git merge -q main >/dev/null 2>&1 || true
echo resolved > r.txt && git add r.txt && git commit -qm "Merge main (conflict resolved)"
expect_eq "reviewed-sha: conflict-resolved merge forces head" "$(sha_for)" "$(git rev-parse HEAD)"
