# Blast radius map, Dograh fork (`VOICE AGENTS - DOGRAH/dograh`, AAlexxis222/dograh)

Kept here until VOZ-N0-00 copies it into the fork as `.github/XPAND_BLAST_RADIUS.md`.
Measured 2026-10-04 on `feat/foundations-base` @ `91d30a9a` (non-test `.py` importers under `api/`).
VOZ-N0-00 (before G0) copies it as is; **VOZ-GATE-G0 re-measures it** (the upstream merge touches ~453 files).

Importer count is NOT the only axis. In this repo the deciding question is: **is it on the path of every live call?**

## Trunk

`review-profile.json` `risk_paths` is the machine copy of this table, read by the review tool at every PR base: change both in the same commit.

| Path | Why |
|---|---|
| `pipecat/` (submodule → dograh-hq/pipecat) | every call runs through it; a bump is a dependency upgrade of the whole media path |
| `api/services/pipecat/run_pipeline.py`, `pipeline_builder.py`, `event_handlers.py`, `service_factory.py` | call hot path, every call (importers 8 / 2 / 1 / 3); `pipeline_builder` also feeds `workflow/text_chat_runner.py`, so text chat too |
| `api/services/pipecat/openai_llm.py`, `openai_efforts.py` | build every OpenAI and Atlas Cloud LLM and translate the plan for OpenRouter, Azure and Dograh (called from `service_factory.py`); `openai_responses.py` stays leaf while `api: responses` is opt-in |
| `api/services/pipecat/service_tuning_specs.py` | holds the build gates (`*_AVAILABLE`, l.886-1017) and is a G0 conflict file |
| `api/db/models` (65) + `api/alembic/versions/` | schema; a migration is a **one-way door** (one head in the fork, `a7c2e9d41b06`; it becomes two heads against upstream at G0: VOZ-UNV-B9-06) |
| `api/services/workflow/` (64), `telephony/` (56), `configuration/` (35), `auth/` (24), `api/utils` (49) | shared by most routes; telephony = real calls to real people |
| `api/routes/telephony.py` | exposed telephony route (call webhooks and media websocket on top of `api/services/telephony/`): always high risk |
| `docker-compose*.yaml`, `deploy/`, `nginx/`, `api/Dockerfile` | deploy and capacity (`max_conns`, grace periods) |
| `.github/workflows/` | CI, release automation |
| `scripts/xpand/call_entrypoint.sh`, `scripts/xpand/require_db_head.sh` | PID 1 of the `call` role, and the start gate of every cell role |
| `api/services/runtime/cell_startup.py` | the start check of every cell role (api lifespan, and `python -m` from `require_db_head.sh`); a wrong refusal stops every role of the cell |
| `api/services/runtime/cell_state.py` | `parse_call_k_p`, the `CALL_K_P` check `cell_startup.py` refuses to start on (a wrong raise stops every role of the cell), and the calls-per-pod and draining state the health endpoint reports (`api/routes/main.py`) |
| `api/services/runtime/durations.py` | the cell durations check `require_db_head.sh` runs first at the start of every cell role (`python -m api.services.runtime.durations`: a wrong raise stops every role of the cell), and the single source of slot TTLs (`api/services/call_concurrency/`), ring timeouts of every telephony provider and the workflow duration ceiling |
| `api/services/security/credential_documents.py` (the rule: leaves, AAD, encoding, fail closed) + `api/db/credentials_at_rest.py` (its ORM wiring, registered by `api/db/__init__.py`) + `api/services/security/secret_surfaces.py` | ORM events on every model with a registered secret column (workflows, definitions, runs, org/user configurations, telephony configurations, tools, credentials, webhook deliveries, integrations, templates): every load decrypts, every flush encrypts. A bug is an outage (a call or route that cannot read its credentials) or a leak (plaintext written); a change to the registry or to the leaf encoding/AAD after tokens exist strands them (needs a migration). Raw SQL on those columns must seal/unseal: `api/tests/test_credentials_at_rest_guard.py` |
| `api/services/security/secret_walk.py` | the leaf walk (`iter_leaves`, `iter_concrete_leaves`, `value_at`) from which `credential_documents.py` decides which leaves are encrypted and with which AAD path (`credential_box.aad_for`); masking and the 422 writer in `api/services/configuration/` read it too. A change to how a pattern reaches its leaves (`*`, `**`, list broadcast) writes plaintext or strands existing tokens (needs a migration) |
| `api/services/security/egress_guard.py`, `api/services/security/egress_flag.py` | the egress URL guard of every tool (custom HTTP tools, transfer resolver, MCP servers, the pre-call fetch) and of every service URL a tenant sets (model providers, STT/TTS, embeddings: checked at call start, at configuration save and by the validator's key check) on the call hot path, and the flag rule the cell start check shares (`egress_flag.py`, stdlib only); with `URL_GUARD_ENFORCE=true` (every cell) a wrong block breaks every tool call and every call's service construction (HTTP 400 at call start), a wrong allow reopens SSRF to IMDS / the compose network (VOZ-AC-B5-91) |
| `api/services/workflow/mcp_tool_session.py` | owns the MCP connection itself (the SDK's streamable-HTTP transport + `ClientSession` on the httpx2 client `egress_guard` builds, no longer Pipecat's `MCPClient`): every MCP tool at call start, MCP discovery and refresh; a regression drops the MCP tools of every call (sessions degrade, the call continues) |
| `api/services/security/redaction.py`, `api/logging_config.py`, `api/services/security/credential_box.py` | every process, every log line: a regression either leaks a secret or phone number to the sinks or slows the call loop; `credential_box` is imported at load by `redaction.py` and `cell_startup.py`, so a broken import (a missing `cryptography` wheel) stops every process |
| `api/errors/failure.py` | `redact_credentials` and `fold_for_gate`, the credential-shape rules `redaction.py` applies to every log line: a regression leaks a secret to the sinks or slows the call loop |

## Leaf (while additive and gated)

- New `adapters/<provider>/`, new engines running in **shadow**, new presets that are not the platform default (once those mechanisms exist, see below).
- `evals/`, `docs/`, `api/tests/`, `scripts/xpand/` (except the two trunk files above).

## Gating mechanisms

**Exist today** (verified at `91d30a9a`; use these, don't invent flags):
- Build gates: `*_AVAILABLE` constants in `service_tuning_specs.py` l.886-1017.
- Per-workflow config: `workflow_configurations`.

**Arrive with spec v4. NOT in the code yet.** A change that claims protection from one of these before its task has merged counts as **ungated**:
- Presets `api/services/configuration/presets/` + `PLATFORM_DEFAULT_PRESET` (B1 VOZ-AC-B1-02/61/64).
- Shadow → activation; promotion as a one-line PR with `run_id` and Alexis's OK (B9).
- Release canary across cells + rollback (B5-45 / B5-46).
