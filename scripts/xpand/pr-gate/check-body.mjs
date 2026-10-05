// Validates a PR description against .github/pull_request_template.md (skill `pr-ready`).
// Shared by the CI job (.github/workflows/xpand-pr-body.yml, job pr-body).
// CLI: node check-body.mjs <body-file> [--no-attribution] [--head-sha <sha>]  → exit 1 + one line per problem.

const SECTIONS = ['Blast radius', 'Gate / rollback', 'Proof', 'Confidence', 'Review']

/** What GitHub renders as content: no HTML comments, no fenced code blocks. */
const uncommented = (body) => body.replace(/^﻿/, '').replace(/<!--[\s\S]*?(-->|$)/g, '')
const visible = (body) => uncommented(body)
  .replace(/^ {0,3}(```|~~~)[\s\S]*?(^ {0,3}\1.*$|(?![\s\S]))/gm, '') // an unclosed fence runs to the end
  .replace(/^( {4,}|\t).*$/gm, '') // indented code block

const headingName = (line) => line.match(/^\s{0,3}##\s+(.*?)[\s#:]*$/)?.[1]?.trim().toLowerCase()

/** All `## <name>` blocks (text up to the next `## `), template bullets stripped. */
function sections(text, name) {
  const lines = text.split(/\r?\n/)
  const out = []
  lines.forEach((l, i) => {
    if (headingName(l) !== name.toLowerCase()) return
    const rest = lines.slice(i + 1)
    const end = rest.findIndex((r) => /^\s{0,3}##\s/.test(r))
    out.push((end < 0 ? rest : rest.slice(0, end)).join('\n').replace(/^\s*-\s*(trunk|leaf):\s*$/gim, '').trim())
  })
  return out
}

/** → list of problems; empty list = valid. headSha: when given, `REVIEWED: <sha>` must match it. */
export function checkBody(body, { noAttribution = false, headSha = null } = {}) {
  const problems = []
  const text = visible(body ?? '')
  for (const name of SECTIONS) {
    const found = sections(text, name)
    if (found.length === 0) problems.push(`missing section "## ${name}"`)
    else if (found.length > 1) problems.push(`section "## ${name}" appears ${found.length} times`)
    else if (!found[0]) problems.push(`section "## ${name}" is empty`)
  }
  const review = sections(text, 'Review')[0] ?? ''
  // Exactly one verdict line and one sha line, each on its own line (bold allowed), outside code.
  const verdicts = [...review.matchAll(/^ {0,3}\**VERDICT:?\**:?\s*(SAFE TO MERGE|MERGE AFTER FIXES|DO NOT MERGE)\b/gim)].map((m) => m[1].toUpperCase())
  const shas = [...review.matchAll(/^ {0,3}\**REVIEWED:?\**:?\s*([0-9a-f]{7,40})\b/gim)].map((m) => m[1])
  const hint = /VERDICT:/i.test(uncommented(body ?? '')) ? ' (VERDICT/REVIEWED must be plain lines, not inside a code block)' : ''
  const verdict = verdicts[0]
  if (review && verdicts.length === 0) problems.push(`"## Review" has no line "VERDICT: SAFE TO MERGE" from the adversarial review${hint}`)
  else if (verdicts.length > 1) problems.push(`"## Review" has ${verdicts.length} VERDICT lines: keep only the one for the head commit`)
  else if (verdict && verdict !== 'SAFE TO MERGE') problems.push(`review verdict is ${verdict}: fix the findings and re-run the review until SAFE TO MERGE`)
  const reviewed = shas.length === 1 ? shas[0] : null
  if (review && shas.length === 0) problems.push(`"## Review" has no line "REVIEWED: <commit sha>" naming the commit the review covered${hint}`)
  else if (shas.length > 1) problems.push(`"## Review" has ${shas.length} REVIEWED lines: keep only one`)
  else if (reviewed && headSha && !headSha.startsWith(reviewed)) {
    problems.push(`review covered ${reviewed}, but the PR head is ${headSha.slice(0, 12)}: re-run the review on the new commits and update the line`)
  }
  if (noAttribution && /claude|anthropic/i.test(body ?? '')) problems.push('mentions Claude/Anthropic (forbidden in the Dograh fork, VOZ-AC-B9-27)')
  return problems
}

if (process.argv[1]?.endsWith('check-body.mjs')) {
  const { readFileSync } = await import('node:fs')
  const args = process.argv.slice(2)
  const file = args[0]
  if (!file) { console.error('usage: node check-body.mjs <body-file> [--no-attribution] [--head-sha <sha>]'); process.exit(2) }
  const shaAt = args.indexOf('--head-sha')
  const problems = checkBody(readFileSync(file, 'utf8'), {
    noAttribution: args.includes('--no-attribution'),
    headSha: shaAt > 0 ? args[shaAt + 1] : null,
  })
  for (const p of problems) console.error(`pr-body: ${p}`)
  process.exit(problems.length ? 1 : 0)
}
