#!/usr/bin/env bash
# scripts/xpand/check_no_claude_attribution.sh <base-ref>, VOZ-AC-B9-27: XPAND-authored commits only
set -euo pipefail
base="${1:?usage: check_no_claude_attribution.sh <base-ref>}"

# Not inside `$(... || true)`: an unknown ref must fail closed, not read as "0 hits".
if ! log="$(git log --format=%B "${base}..HEAD" 2>&1)"; then
  echo "code=attribution_check_failed where=${base}..HEAD reason=git log failed: ${log} hint=fetch the base ref (fetch-depth: 0) and retry" >&2
  exit 1
fi
hits="$(grep -ciE 'claude|anthropic' <<<"$log" || true)"
if [[ "$hits" != "0" ]]; then
  echo "code=claude_attribution_found where=${base}..HEAD reason=${hits} line(s) mention claude/anthropic hint=reword the commits (Alexis runs filter-branch with !)" >&2
  exit 1
fi
