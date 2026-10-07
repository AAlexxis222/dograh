## Checklist

- [ ] I have read and followed the [contributing guidelines](https://github.com/dograh-hq/dograh/blob/main/CONTRIBUTING.md).

---

<!-- XPAND fork section (VOZ-N0-00, B9 §8.3). Checked by .github/workflows/xpand-pr-body.yml (job pr-body). -->

## What / why
<!-- 2-3 lines. Link the spec/plan task. The whole description must be shorter than the diff. -->

## Blast radius
<!-- See the blast-radius map: .github/XPAND_BLAST_RADIUS.md -->
- trunk:
- leaf:

## Gate / rollback
<!-- flag/config + default OFF | revert-only | ONE-WAY DOOR: what and why -->

## Proof
<!-- commands run + result (build/lint/test), runtime logs, screenshots. Old path with gate OFF, new path with gate ON.
If the PR touches migrations: MIGRATION-REHEARSAL: <test or command> → <result> (DB seeded with every row shape).
If it is part of a planned merge list sharing files with another PR: MERGE-ORDER: <order> → <result per step>. -->

## Confidence
<!-- high / medium / low per area, one reason each -->

## Review
<!-- Required, checked by the pr-body gate: the two lines below, as adversarial-reviewer prints them, for the PR head commit,
plus the principles-reviewer line.
VERDICT: SAFE TO MERGE
REVIEWED: <head commit sha>
PRINCIPLES: <n> MAJOR / <n> MINOR → <fixed | accepted: why>   (or: PRINCIPLES: n/a (no code))
Then: what was fixed / pushed back. -->

## XPAND obligations (B9 §8.3)

- [ ] No AI-tool attribution in commits or in this description (VOZ-AT-B9-05)
- [ ] Code-quality guides and engineering principles applied to the task
- [ ] Every `pin`/`fork` citation this PR implements re-anchored to the reconciled tree before touching it (re-anchor table below); "Informe" citations confirmed by reading
- [ ] Capability-table test: every pinned knob has a row for the resolved model; loader fails on a missing native field; a date in two places is an error (VOZ-AT-B1-01, -23)
- [ ] Deprecations test: no default points to a past-dated or `legacy` model (VOZ-AT-B1-13)
- [ ] Data-boundary test: `completion_gate: jev` / `backchannel_gate: jev` outside the LAB allowlist is a 422 (VOZ-AT-B1-22)
- [ ] Egress boundary: a run only opens sockets to `resolved_services` + annex III + the cell S3 (VOZ-AT-B6-04)
- [ ] No plaintext secret in DB, logs, `effective_configurations`, `resolution_report` or responses (VOZ-AT-B6-05, VOZ-AT-B1-27)
- [ ] Telemetry off in a cell: `ENABLE_TELEMETRY` absent/`false`, 0 sockets to PostHog/Sentry without DSN (VOZ-AT-B6-07)
- [ ] Import boundary: no `engines/`/`turns/` module imports `pipecat.services.<provider>` (VOZ-AT-B2-01, VOZ-AT-B1-24)
- [ ] ORT gate: no `InferenceSession(` without `ort_session_options()` (VOZ-AT-B3-08)
- [ ] Module-level state: 0 new dicts/sets in `api/` outside the allowlist (VOZ-AT-B3-03)
- [ ] Durations: `durations.py` is the only source; no hand-written value in Compose/Helm/scripts (VOZ-AT-B3-19)
- [ ] Load profile row in `capacity/profiles.yaml` for any adapter with `cpu_cost_class` other than `remote`
- [ ] Trace lint: no `warning` without a TRACE-1 event; no `ERROR` log without `code` (VOZ-AT-B7-32, VOZ-AT-B9-07)
- [ ] `--selftest` of the 25 probes + `bench/load` green in `xva` CI (VOZ-AT-B8-01)
- [ ] Non-empty `cap_rows_affected` in a `live` run: this PR updates the CAP with `source: [M] <run_id>` (VOZ-AT-B8-04)
- [ ] No preset change without an arena `run_id` and a resolved conflict
- [ ] `provider` PR: conformance summary with the five probes (VOZ-AT-B8-06)
- [ ] `cells/<client_id>.yaml` schema, only digests in the overlay, `tofu plan` in the PR (VOZ-AT-B5-05/06/07)
- [ ] Expand-only migrations (linter), never run at startup (VOZ-AC-B5-43/44)
- [ ] DRY of blocks: this PR does not duplicate a mechanism another block owns
- [ ] Inherited from v3 §12: `./scripts/format.sh`, `openapi.json`, `types.gen.ts`, SDK generation, fork CI green, `pytest` from `api/` with `.env.test`, every new test mutation-verified
