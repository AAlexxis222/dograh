// node --test scripts/xpand/pr-gate/pr-gate.test.mjs   (a directory argument does not work on Node ≥ 21)
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { checkBody } from './check-body.mjs'

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
test('two verdict lines fail, whatever the order', () => {
  assert.match(checkBody(good.replace('REVIEWED:', 'VERDICT: DO NOT MERGE\nREVIEWED:')).join(), /2 VERDICT lines/)
})
test('attribution only forbidden when asked', () => {
  const b = `${good}\nGenerated with ${['Cla', 'ude'].join('')} Code` // built at runtime: no literal mention in the fork tree
  assert.deepEqual(checkBody(b), [])
  assert.match(checkBody(b, { noAttribution: true }).join(), /Claude/)
})
