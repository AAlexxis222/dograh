"""Key schema of the call-concurrency slots in Redis (VOZ-AC-B3-32, -65).

One canonical member for every slot set: the ``attempt_id`` of the admission (``new_attempt_id``), the same string
in the org set, the scope set and the fleet set, so a release built from the same attempt can never miss a key
because of a format mismatch.

Score of every slot member = its expiry in Redis's clock (``TIME + ttl``): ``TIME + pending_ttl_s`` (inbound) or
``TIME + outbound_pending_ttl_s`` (outbound, it rings first) while the admission waits for a worker,
``TIME + slot_ttl`` once the worker claims it and at every renewal. A member whose score is <= ``TIME`` is expired:
writers purge it, readers do not count it.

The slot sets live under the ``v2`` namespace. The previous code scored members with the acquisition time on a process
clock in the un-versioned keys; while both versions run (a rolling deploy) these LEGACY keys are still counted with
the old rule (live while score > TIME - slot_ttl), read-only, inside the same scripts, and the release funnel also
removes a legacy call it ends. They die on their own EXPIRE.
Follow-up VOZ-N0-21-F1: drop every ``legacy_*`` read one release after this one.

Assumption: one single-node Redis per cell. The scripts touch several keys at once (v2 and legacy sets of an org,
the fleet set, the mapping and the ``sem:`` keys named inside it), which Redis Cluster would only allow within one
hash slot; the existing keys carry no hash tags, so none are added here. Moving to a cluster means tagging every key
of an org (both versions) with the same ``{org}`` tag and rethinking the fleet set.

Keys, and every reader / writer of them (all in ``rate_limiter.py``):

- ``org_key(org_id)``: ZSET ``concurrent_calls:v2:<org_id>``.
  Writers: ``acquire_slot`` (purge + add pending), ``claim_slot`` / ``renew_slot`` (re-add claimed),
  ``release_slot`` (remove). Readers: ``get_concurrent_count``, ``acquire_slot``, ``claim_slot`` (purge + count).
- ``scope_key(scope)``: ZSET ``concurrent_calls:v2:<scope>`` (e.g. ``campaign:<id>``), same writers; read by
  ``acquire_slot`` and ``claim_slot``. Scope values never collide with org keys: they always carry a ``<kind>:``
  prefix.
- ``fleet_key()``: ZSET ``concurrent_calls_fleet:v2``, the fleet-wide mirror of every org slot (autoscaling
  signal). Writers: ``acquire_slot`` (purge + add), ``claim_slot`` / ``renew_slot``, ``release_slot``.
  Reader: ``get_fleet_concurrent_count`` (count of members with score > ``TIME``, no write).
- ``legacy_org_key`` / ``legacy_scope_key`` / ``rate_limiter.FLEET_CONCURRENT_KEY`` (fleet member
  ``legacy_fleet_member``): the pre-v2 sets. Read by every count above; written only by ``release_slot`` (remove).
- ``mapping_key(workflow_run_id)``: HASH ``workflow_slot_mapping:<run>`` (shared by both versions) with ``org_id``,
  ``slot_id`` (the attempt_id), optional ``scope_key``, the admission limits ``max_concurrent`` /
  ``scope_max_concurrent`` (for a late claim), and one ``sem:<zset key>`` field per provider semaphore member of
  the attempt (none are written yet: the semaphores arrive with the B1 capacity rows; ``release_slot`` already
  removes them). Writers: ``store_workflow_slot_mapping[_if_absent]`` (bind), ``claim_slot`` / ``renew_slot``
  (refresh TTL, re-create after a Redis restart), ``release_slot`` (delete with its slot). Readers:
  ``get_workflow_slot_mapping``, ``claim_slot``, ``reconcile_workflow_slot_mapping``.
"""

import uuid

VERSION = "v2"


def new_attempt_id() -> str:
    """The admission id: the one member every slot key of the attempt carries."""
    return uuid.uuid4().hex


def org_key(organization_id: int) -> str:
    return f"concurrent_calls:{VERSION}:{organization_id}"


def scope_key(scope: str | None) -> str:
    """Empty when the slot has no scope: the Lua scripts skip an empty key."""
    return f"concurrent_calls:{VERSION}:{scope}" if scope else ""


def fleet_key() -> str:
    return f"concurrent_calls_fleet:{VERSION}"


def legacy_org_key(organization_id: int) -> str:
    return f"concurrent_calls:{organization_id}"


def legacy_scope_key(scope: str | None) -> str:
    return f"concurrent_calls:{scope}" if scope else ""


def legacy_fleet_member(organization_id: int, attempt_id: str) -> str:
    return f"{organization_id}:{attempt_id}"


def mapping_key(workflow_run_id: int | None) -> str:
    """Empty when the slot is not bound to a run yet: the Lua scripts skip an empty key."""
    return f"workflow_slot_mapping:{workflow_run_id}" if workflow_run_id else ""
