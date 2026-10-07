#!/usr/bin/env bash
# source: xpand plugin tools/pr-gate @5449572
# bash scripts/xpand/pr-gate/reviewed-sha.test.sh  — scenarios from the 3rd adversarial review of the gate.
set -eo pipefail
script="$(cd "$(dirname "$0")" && pwd)/reviewed-sha.sh"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/repo" && cd "$tmp/repo" && git init -q -b main . && git config user.email t@t && git config user.name t && git config commit.gpgsign false
c() { echo "$2" >> "$1"; git add -A; git commit -qm "$3"; }
c base.txt 1 base1
git checkout -qb feat; c feat.txt reviewed f1
R=$(git rev-parse HEAD)
printf '## Review\nREVIEWED: %s\n' "${R:0:12}" > "$tmp/body.md"
fails=0
expect() { # <label> <want: reviewed|head>
  local head got want; head=$(git rev-parse HEAD); got=$("$script" "$tmp/body.md" "$head" main)
  [ "$2" = reviewed ] && want="$R" || want="$head"
  if [ "$got" = "$want" ]; then echo "PASS $1"; else echo "FAIL $1 (got $got)"; fails=$((fails+1)); fi
}
expect "reviewed head itself" reviewed
git checkout -q main; c base.txt 2 base2; git checkout -q feat
git merge -q --no-edit main; expect "clean update-branch keeps the review" reviewed
c feat.txt unreviewed f2; expect "own commit after review needs a new review" head
git reset -q --hard HEAD~1
git checkout -q main; c base.txt 3 base3; git checkout -q feat
git merge -q --no-commit main; echo EVIL >> feat.txt; git add -A; git commit -qm "Merge main"
expect "merge with extra edits needs a new review" head
git reset -q --hard HEAD~1
git checkout -qb side main; c side.txt unreviewed s1; git checkout -q feat
git merge -q --no-ff --no-edit side; expect "merge of a non-base branch needs a new review" head
# Back on the clean update-branch merge, where the stale sha R would be accepted if it were read.
git reset -q --hard HEAD~1
H=$(git rev-parse HEAD)
printf '```
REVIEWED: %s
```
## Review
VERDICT: SAFE TO MERGE
REVIEWED: %s
' "${R:0:12}" "${H:0:12}" > "$tmp/body2.md"
got=$("$script" "$tmp/body2.md" "$H" main)
if [ "$got" = "$H" ]; then echo "PASS REVIEWED in a code block is ignored"; else echo "FAIL REVIEWED in a code block is ignored (got $got)"; fails=$((fails+1)); fi
exit "$fails"
