// Validates a PR description against .github/PULL_REQUEST_TEMPLATE.md. Used by .github/workflows/xpand-pr-body.yml.
// source: xpand plugin tools/pr-gate @c0477d6
// CLI: node check-body.mjs <body-file> [--no-attribution] [--head-sha <sha>] [--changed-files <file>] [--migration-prefix <prefix>]...
//      node check-body.mjs <body-file> --print-reviewed   → prints the REVIEWED sha (or nothing), exit 0
//      validation exits 1 + one line per problem.

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

const REVIEWED_RE = /^ {0,3}\**REVIEWED:?\**:?\s*([0-9a-f]{7,40})\b/gim

/** The sha on the visible "## Review" REVIEWED line, or null unless there is exactly one. */
export function reviewedSha(body) {
  const review = sections(visible(body ?? ''), 'Review')[0] ?? ''
  const shas = [...review.matchAll(REVIEWED_RE)].map((m) => m[1])
  return shas.length === 1 ? shas[0] : null
}

const PRINCIPLES_RE = /^ {0,3}\**PRINCIPLES\**:\**[ \t]*(.*)$/gim
const PRINCIPLES_OK = /^(n\/a \(no code\)|\d+ MAJOR \/ \d+ MINOR[ \t]*(→|->)[ \t]*(fixed\b.*|accepted:[ \t]*\S.*))$/i
const REHEARSAL_RE = /^ {0,3}\**MIGRATION-REHEARSAL:?\**:?[ \t]*\S.*?[ \t]*(→|->)[ \t]*\S.*$/im

function principlesProblems(review) {
  if (!review) return []
  const lines = [...review.matchAll(PRINCIPLES_RE)].map((m) => m[1].trim())
  if (lines.length === 0) return ['"## Review" has no line "PRINCIPLES: <n> MAJOR / <n> MINOR → <fixed | accepted: why>" (or "PRINCIPLES: n/a (no code)")']
  if (lines.length > 1) return [`"## Review" has ${lines.length} PRINCIPLES lines: keep only one`]
  return PRINCIPLES_OK.test(lines[0]) ? [] : [`PRINCIPLES line "${lines[0]}" must read "<n> MAJOR / <n> MINOR → <fixed | accepted: why>" or "n/a (no code)"`]
}

function rehearsalProblems(text, { changedFiles = null, migrationPrefixes = [] }) {
  const proof = sections(text, 'Proof')[0] ?? ''
  const touches = (changedFiles ?? []).some((f) => migrationPrefixes.some((p) => f.startsWith(p)))
  if (!touches || REHEARSAL_RE.test(proof)) return []
  return [`this PR touches migrations (${migrationPrefixes.join(', ')}): "## Proof" needs "MIGRATION-REHEARSAL: <test or command> → <result>" run on a database seeded with every row shape the migration reads`]
}

/** → list of problems; empty list = valid. headSha: when given, `REVIEWED: <sha>` must match it. changedFiles + migrationPrefixes: a migration touch requires MIGRATION-REHEARSAL. */
export function checkBody(body, options = {}) {
  const { noAttribution = false, headSha = null } = options
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
  const shas = [...review.matchAll(REVIEWED_RE)].map((m) => m[1])
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
  problems.push(...principlesProblems(review))
  problems.push(...rehearsalProblems(text, options))
  if (noAttribution && /claude|anthropic/i.test(body ?? '')) problems.push('mentions Claude/Anthropic (forbidden in the Dograh fork, VOZ-AC-B9-27)')
  return problems
}

if (process.argv[1]?.endsWith('check-body.mjs')) {
  const { readFileSync } = await import('node:fs')
  const args = process.argv.slice(2)
  const file = args[0]
  if (!file) { console.error('usage: node check-body.mjs <body-file> [--print-reviewed | [--no-attribution] [--head-sha <sha>] [--changed-files <file>] [--migration-prefix <prefix>]...]'); process.exit(2) }
  const fail = (msg) => { console.error(`pr-body: ${msg}`); process.exit(2) }
  if (args.includes('--print-reviewed')) {
    const sha = reviewedSha(readFileSync(file, 'utf8'))
    if (sha) console.log(sha)
    process.exit(0)
  }
  const valueAt = (i) => (args[i + 1] === undefined || args[i + 1].startsWith('--') ? fail(`${args[i]} needs a value`) : args[i + 1])
  const valueOf = (flag) => { const i = args.indexOf(flag); return i > 0 ? valueAt(i) : null }
  const changed = valueOf('--changed-files')
  let changedFiles = null
  if (changed) {
    try { changedFiles = readFileSync(changed, 'utf8').split(/\r?\n/).filter(Boolean) } catch { fail(`cannot read --changed-files ${changed}`) }
  }
  const problems = checkBody(readFileSync(file, 'utf8'), {
    noAttribution: args.includes('--no-attribution'),
    headSha: valueOf('--head-sha'),
    changedFiles,
    migrationPrefixes: args.flatMap((a, i) => (a === '--migration-prefix' ? [valueAt(i)] : [])),
  })
  for (const p of problems) console.error(`pr-body: ${p}`)
  process.exit(problems.length ? 1 : 0)
}
