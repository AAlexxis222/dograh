# Blast radius map, Dograh fork (`VOICE AGENTS - DOGRAH/dograh`, AAlexxis222/dograh)

Kept here until VOZ-N0-00 copies it into the fork as `.github/XPAND_BLAST_RADIUS.md`.
Measured 2026-10-04 on `feat/foundations-base` @ `91d30a9a` (non-test `.py` importers under `api/`).
VOZ-N0-00 (before G0) copies it as is; **VOZ-GATE-G0 re-measures it** (the upstream merge touches ~453 files).

Importer count is NOT the only axis. In this repo the deciding question is: **is it on the path of every live call?**

## Trunk

| Path | Why |
|---|---|
| `pipecat/` (submodule → dograh-hq/pipecat) | every call runs through it; a bump is a dependency upgrade of the whole media path |
| `api/services/pipecat/run_pipeline.py`, `pipeline_builder.py`, `event_handlers.py`, `service_factory.py` | call hot path, every call (importers 8 / 2 / 1 / 3); `pipeline_builder` also feeds `workflow/text_chat_runner.py`, so text chat too |
| `api/services/pipecat/service_tuning_specs.py` | holds the build gates (`*_AVAILABLE`, l.886-1017) and is a G0 conflict file |
| `api/db/models` (65) + `api/alembic/versions/` | schema; a migration is a **one-way door** (one head in the fork, `a7c2e9d41b06`; it becomes two heads against upstream at G0: VOZ-UNV-B9-06) |
| `api/services/workflow/` (64), `telephony/` (56), `configuration/` (35), `auth/` (24), `api/utils` (49) | shared by most routes; telephony = real calls to real people |
| `docker-compose*.yaml`, `deploy/`, `nginx/`, `api/Dockerfile` | deploy and capacity (`max_conns`, grace periods) |
| `.github/workflows/` | CI, release automation |
| `scripts/xpand/call_entrypoint.sh`, `scripts/xpand/require_db_head.sh` | PID 1 of the `call` role, and the start gate of every cell role |
| `api/services/runtime/cell_startup.py` | the start check of every cell role (api lifespan, and `python -m` from `require_db_head.sh`); a wrong refusal stops every role of the cell |
| `api/services/workflow/tools/url_guard.py` | the URL guard of every tool egress (custom HTTP tools, transfer resolver, MCP servers) on the call hot path; with `URL_GUARD_ENFORCE=true` (every cell) a wrong block breaks every tool call, a wrong allow reopens SSRF to IMDS / the compose network (VOZ-AC-B5-91) |
| `api/services/workflow/mcp_tool_session.py` | owns the MCP connection itself (the SDK's streamable-HTTP transport + `ClientSession` on an httpx2 client it builds, no longer Pipecat's `MCPClient`): every MCP tool at call start, MCP discovery and refresh; a regression drops the MCP tools of every call (sessions degrade, the call continues) |
| `api/services/security/redaction.py`, `api/logging_config.py`, `api/services/security/credential_box.py` | every process, every log line: a regression either leaks a secret or phone number to the sinks or slows the call loop; `credential_box` is imported at load by `redaction.py` and `cell_startup.py`, so a broken import (a missing `cryptography` wheel) stops every process |

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
