// source: xpand plugin tools/pr-gate @c0477d6 (the local-hook tests of the source are not ported: hook.mjs is not part of the fork)
// node --test scripts/xpand/pr-gate/pr-gate.test.mjs   (a directory argument does not work on Node ≥ 21)
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { checkBody, reviewedSha } from './check-body.mjs'

const template = readFileSync(new URL('../../../.github/PULL_REQUEST_TEMPLATE.md', import.meta.url), 'utf8')
const good = `## What / why
Bump X.

## Blast radius
- trunk: cli/factory-core/pipeline
- leaf: docs

## Gate / rollback
revert-only

## Proof
go test ./... → ok

## Confidence
high: covered by tests

## Review
VERDICT: SAFE TO MERGE
REVIEWED: abc1234
PRINCIPLES: 0 MAJOR / 1 MINOR → accepted: naming kept for upstream parity
`

// --- validator
test('a filled body passes', () => assert.deepEqual(checkBody(good), []))
test('CRLF and bold verdict pass', () => assert.deepEqual(checkBody(good.replace(/\n/g, '\r\n').replace('VERDICT:', '**VERDICT:**')), []))
test('the untouched template fails on every section', () => {
  assert.equal(checkBody(template).filter((p) => p.endsWith('is empty')).length, 5)
})
test('missing verdict fails', () => assert.match(checkBody(good.replace(/VERDICT:.*/, 'looked fine')).join(), /no line "VERDICT/))
test('verdict must be its own line', () => {
  assert.match(checkBody(good.replace('VERDICT: SAFE TO MERGE', 'reviewer did NOT say VERDICT: SAFE TO MERGE')).join(), /no line "VERDICT/)
})
test('only SAFE TO MERGE passes', () => {
  assert.match(checkBody(good.replace('SAFE TO MERGE', 'DO NOT MERGE')).join(), /DO NOT MERGE/)
  assert.match(checkBody(good.replace('SAFE TO MERGE', 'MERGE AFTER FIXES')).join(), /MERGE AFTER FIXES/)
})
test('missing section fails', () => assert.match(checkBody(good.replace('## Proof', '## Evidence')).join(), /missing section "## Proof"/))
test('a second Review section fails', () => assert.match(checkBody(`${good}\n## Review\nVERDICT: DO NOT MERGE\n`).join(), /appears 2 times/))
test('a body hidden in an HTML comment fails', () => assert.ok(checkBody(`<!--\n${good}\n-->`).length >= 5))
test('headings inside a code fence do not count', () => assert.ok(checkBody(`\`\`\`\n${good}\n\`\`\``).length >= 5))
test('REVIEWED sha must be present and match the head when given', () => {
  assert.match(checkBody(good.replace(/REVIEWED:.*/, '')).join(), /no line "REVIEWED/)
  assert.deepEqual(checkBody(good, { headSha: 'abc1234def' }), [])
  assert.match(checkBody(good, { headSha: 'fff0000aaa' }).join(), /re-run the review/)
})
test('an unclosed fence hides the rest', () => assert.ok(checkBody(`\`\`\`\n${good}`).length >= 5))
test('a verdict in an indented code block does not count', () => {
  assert.match(checkBody(good.replace('VERDICT: SAFE TO MERGE', '    VERDICT: SAFE TO MERGE')).join(), /no line "VERDICT/)
})
test('a fenced verdict gets a hint', () => {
  assert.match(checkBody(good.replace('VERDICT: SAFE TO MERGE\nREVIEWED: abc1234', '```\nVERDICT: SAFE TO MERGE\nREVIEWED: abc1234\n```\nok')).join(), /not inside a code block/)
})
test('reviewedSha reads only the visible Review line', () => {
  assert.equal(reviewedSha(good), 'abc1234')
  assert.equal(reviewedSha(good.replace('REVIEWED: abc1234', '<!-- REVIEWED: abc1234 -->')), null)
  assert.equal(reviewedSha(`REVIEWED: fff0000\n${good}`), 'abc1234')
  assert.equal(reviewedSha(good.replace('REVIEWED: abc1234', 'REVIEWED: abc1234\nREVIEWED: def5678')), null)
})
test('PRINCIPLES line is required', () =>
  assert.match(checkBody(good.replace(/^PRINCIPLES:.*$/m, '')).join(), /no line "PRINCIPLES:/))
test('PRINCIPLES n/a passes', () =>
  assert.deepEqual(checkBody(good.replace(/^PRINCIPLES:.*$/m, 'PRINCIPLES: n/a (no code)')), []))
test('PRINCIPLES with a bad format fails', () =>
  assert.match(checkBody(good.replace(/^PRINCIPLES:.*$/m, 'PRINCIPLES: looked fine')).join(), /PRINCIPLES/))
test('PRINCIPLES outside Review does not count', () =>
  assert.match(checkBody(good.replace(/^PRINCIPLES:.*$/m, '').replace('go test ./... → ok', 'go test ./... → ok\nPRINCIPLES: n/a (no code)')).join(), /no line "PRINCIPLES:/))
const mig = { changedFiles: ['api/alembic/versions/abc_add.py', 'api/x.py'], migrationPrefixes: ['api/alembic/versions/'] }
test('migration change without rehearsal fails', () =>
  assert.match(checkBody(good, mig).join(), /MIGRATION-REHEARSAL/))
test('migration change with rehearsal passes', () =>
  assert.deepEqual(checkBody(good.replace('go test ./... → ok', 'go test ./... → ok\nMIGRATION-REHEARSAL: pytest api/tests/test_migration_g0.py → 12 passed'), mig), []))
test('no migration change needs no rehearsal', () =>
  assert.deepEqual(checkBody(good, { changedFiles: ['api/x.py'], migrationPrefixes: ['api/alembic/versions/'] }), []))
test('rehearsal without a result fails', () =>
  assert.match(checkBody(good.replace('go test ./... → ok', 'go test ./... → ok\nMIGRATION-REHEARSAL: TODO'), mig).join(), /MIGRATION-REHEARSAL/))
test('PRINCIPLES value must be on the same line', () =>
  assert.match(checkBody(good.replace(/^PRINCIPLES:.*$/m, 'PRINCIPLES:\nn/a (no code)')).join(), /PRINCIPLES/))
test('a prose line starting with Principles is not a declaration', () =>
  assert.match(checkBody(good.replace(/^PRINCIPLES:.*$/m, 'Principles reviewer found nothing.')).join(), /no line "PRINCIPLES:/))
test('PRINCIPLES bold label passes', () =>
  assert.deepEqual(checkBody(good.replace(/^PRINCIPLES:/m, '**PRINCIPLES:**')), []))
test('PRINCIPLES arrow needs fixed or accepted: reason', () => {
  assert.match(checkBody(good.replace(/^PRINCIPLES:.*$/m, 'PRINCIPLES: 0 MAJOR / 0 MINOR → TODO')).join(), /PRINCIPLES/)
  assert.deepEqual(checkBody(good.replace(/^PRINCIPLES:.*$/m, 'PRINCIPLES: 1 MAJOR / 0 MINOR → fixed in abc1234')), [])
})
test('rehearsal arrow must be on the same line as the label', () => {
  const withLine = (l) => good.replace('go test ./... → ok', `go test ./... → ok\n${l}`)
  assert.match(checkBody(withLine('MIGRATION-REHEARSAL:\ngo test ./... → ok'), mig).join(), /MIGRATION-REHEARSAL/)
  assert.match(checkBody(withLine('MIGRATION-REHEARSAL: TODO\n→ ok'), mig).join(), /MIGRATION-REHEARSAL/)
  assert.deepEqual(checkBody(withLine('MIGRATION-REHEARSAL: pytest x→12 passed'), mig), [])
})
test('two verdict lines fail, whatever the order', () => {
  assert.match(checkBody(good.replace('REVIEWED:', 'VERDICT: DO NOT MERGE\nREVIEWED:')).join(), /2 VERDICT lines/)
})
test('attribution only forbidden when asked', () => {
  const b = `${good}\nGenerated with ${['Cla', 'ude'].join('')} Code` // built at runtime: no literal mention in the fork tree
  assert.deepEqual(checkBody(b), [])
  assert.match(checkBody(b, { noAttribution: true }).join(), /Claude/)
})
