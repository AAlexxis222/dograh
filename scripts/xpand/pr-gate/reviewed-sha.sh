#!/usr/bin/env bash
# reviewed-sha.sh <body-file> <head-sha> <base-ref>
# Prints the sha the PR body's "REVIEWED:" line must match (check-body.mjs --head-sha):
# the PR head, or the reviewed commit itself when everything after it on the PR's first-parent line
# is a CLEAN merge of the base branch (GitHub "Update branch"). Any own commit, any merge of a
# non-base branch and any merge whose tree differs from git's automatic merge (conflict resolution,
# extra edits) keeps the head → the review must be re-run.
set -eo pipefail
body="$1" head="$2" base="$3"
out="$head"
rev=$(grep -oiE '^ {0,3}\**REVIEWED:?\**:?[[:space:]]*[0-9a-f]{7,40}' "$body" | grep -oiE '[0-9a-f]{7,40}$' | head -1 || true)
if [ -n "$rev" ] && full=$(git rev-parse -q --verify "$rev^{commit}") \
   && git merge-base --is-ancestor "$full" "$head" \
   && [ -z "$(git rev-list --first-parent --no-merges "$full..$head")" ]; then
  ok=1
  for m in $(git rev-list --first-parent --merges "$full..$head"); do
    git merge-base --is-ancestor "$m^2" "$base" || { ok=0; break; }
    tree=$(git merge-tree --write-tree "$m^1" "$m^2" 2>/dev/null | head -1) || { ok=0; break; }
    [ "$tree" = "$(git rev-parse "$m^{tree}")" ] || { ok=0; break; }
  done
  [ "$ok" = 1 ] && out="$full"
fi
echo "$out"
