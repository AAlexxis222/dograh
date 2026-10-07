// source: xpand plugin tools/principles @5449572
// Review system v3 ratchet only (spec 2026-10-06 §3.2). Loaded with `eslint -c`; plugins resolve from the
// node_modules next to this file (installed from package-lock.json in CI), never from the PR.
import comments from '@eslint-community/eslint-plugin-eslint-comments/configs'
import tsParser from '@typescript-eslint/parser'

export default [
  { ignores: ['**/node_modules/**', '**/dist/**', '**/build/**', '**/.next/**'] },
  {
    files: ['**/*.{js,mjs,cjs,jsx,ts,tsx}'],
    languageOptions: { parser: tsParser, ecmaVersion: 'latest', sourceType: 'module', parserOptions: { ecmaFeatures: { jsx: true } } },
    // Off: its messages have ruleId null and would be counted as parse errors.
    linterOptions: { reportUnusedDisableDirectives: 'off' },
    rules: {
      complexity: ['error', 15],
      'max-lines-per-function': ['error', { max: 80, skipBlankLines: true, skipComments: true }],
      'no-empty': ['error', { allowEmptyCatch: false }],
    },
  },
  comments.recommended,
  {
    rules: {
      '@eslint-community/eslint-comments/require-description': 'error',
      // Inline rule-config comments (eslint <rule>: [...] in a block comment) would loosen the limits for a file: only
      // disable/enable comments are allowed here, and the ratchet also rejects them by text.
      '@eslint-community/eslint-comments/no-use': ['error', {
        allow: ['eslint-disable', 'eslint-disable-line', 'eslint-disable-next-line', 'eslint-enable', 'global', 'globals', 'exported'],
      }],
    },
  },
]
