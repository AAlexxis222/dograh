#!/usr/bin/env bash
# scripts/xpand/check_fix_against_g0.sh <pinned-upstream-sha> [<fork-base-ref>], VOZ-AC-B9-03 / VOZ-AT-B9-02
# A FIX that lands before VOZ-GATE-G0 must (a) not touch a G0 conflict file and
# (b) not create a merge conflict with the pinned upstream outside the G0 §3.1 list.
set -euo pipefail
upstream="${1:?usage: check_fix_against_g0.sh <pinned-upstream-sha> [<fork-base-ref>]}"
fork_base="${2:-origin/feat/foundations-base}"
here="$(cd "$(dirname "$0")" && pwd)"
list="$here/g0_conflict_files.txt"   # the 29 files of G0 §3.1 (re-measured), one per line

fail() { # <code> <where> <reason> <hint>
  echo "code=$1 where=$2 reason=$3 hint=$4" >&2
  exit 1
}

# Allowed/forbidden set: the G0 list plus the file VOZ-N0-00 names explicitly; blank lines dropped
# (an empty pattern in `grep -f` would match everything).
allowed="$(sed 's/\r$//; /^$/d' "$list"; echo api/services/pipecat/service_tuning_specs.py)"

# (a) files this branch touches
base="$(git merge-base HEAD "$fork_base")" \
  || fail fix_check_failed "$fork_base" "no merge-base with HEAD" "fetch the fork base branch (fetch-depth: 0)"
touched="$(git diff --name-only "$base"..HEAD)"
bad_touch="$(grep -xF -f <(echo "$allowed") <<<"$touched" || true)"

# (b) conflicts a merge with upstream would create. merge-tree exits 1 both for "conflicts" and for
# some errors (e.g. unknown ref), so resolve the ref first and require a tree oid on the first line.
git rev-parse --verify --quiet "${upstream}^{commit}" >/dev/null \
  || fail fix_check_failed "$upstream" "upstream commit not found" "fetch the pinned upstream sha (git fetch upstream main)"
out="$(git merge-tree --write-tree --name-only HEAD "$upstream" 2>&1)" || true
[[ "$(head -n1 <<<"$out")" =~ ^[0-9a-f]{40,64}$ ]] \
  || fail fix_check_failed "$upstream" "git merge-tree failed: $out" "check the git version (>= 2.38) and the refs"
# Output: tree oid, then conflicted paths, then a blank line, then informational messages.
conflicts="$(sed '1d; /^$/,$d' <<<"$out")"
new="$(grep -vxF -f <(echo "$allowed") <<<"$conflicts" | sed '/^$/d' || true)"

if [[ -n "$new" || -n "$bad_touch" ]]; then
  where="$(echo $new $bad_touch)"
  fail fix_conflicts_with_g0 "$where" "FIX before VOZ-GATE-G0 touches or conflicts on G0 files" "move this task to section B (after VOZ-GATE-G0)"
fi
