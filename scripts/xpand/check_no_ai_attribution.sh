#!/usr/bin/env bash
# scripts/xpand/check_no_ai_attribution.sh <base-ref> [head-ref] [exclude-ref...] | --stdin, VOZ-AC-B9-27: XPAND-authored commits only
# <base-ref>: scans message, author and committer (name+email) of every commit in <base-ref>..<head-ref> (default HEAD),
#             minus anything reachable from an <exclude-ref> (upstream history is not XPAND-authored).
# --stdin:    scans stdin (PR title, branch name) with the same pattern.
set -euo pipefail
arg="${1:?usage: check_no_ai_attribution.sh <base-ref>|--stdin}"
head="${2:-HEAD}" # optional 2nd arg: scan <base-ref>..<head-ref> instead of ..HEAD
excludes=(); for e in "${@:3}"; do excludes+=("^$e"); done # optional 3rd+ args: refs whose history is not scanned
pattern='claude|anthropic'

if [[ "$arg" == "--stdin" ]]; then
  where="stdin"; text="$(cat)"
# Not inside `$(... || true)`: an unknown ref must fail closed, not read as "0 hits".
elif ! text="$(git log --format='%an%n%ae%n%cn%n%ce%n%B' "${arg}..${head}" ${excludes[@]+"${excludes[@]}"} 2>&1)"; then
  echo "code=attribution_check_failed where=${arg}..${head} reason=git log failed: ${text} hint=fetch the base ref (fetch-depth: 0) and retry" >&2
  exit 1
else
  where="${arg}..${head}"
fi
hits="$(grep -ciE "$pattern" <<<"$text" || true)"
if [[ "$hits" != "0" ]]; then
  echo "code=ai_attribution_found where=${where} reason=${hits} line(s) in message/author/committer/title/branch match the forbidden vendor pattern hint=reword the commits or rename (Alexis runs filter-branch with !)" >&2
  exit 1
fi
