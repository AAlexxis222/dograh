"""Key schema of the call-concurrency slots in Redis (VOZ-AC-B3-32, -65).

One canonical member for every slot set: the ``attempt_id`` of the admission (``new_attempt_id``), the same string
in the org set, the scope set and the fleet set, so a release built from the same attempt can never miss a key
because of a format mismatch.

Score of every slot member = its expiry in Redis's clock (``TIME + ttl``): ``TIME + pending_ttl_s`` while the
admission waits for a worker, ``TIME + slot_ttl`` once the worker claims it and at every renewal. A member whose
score is <= ``TIME`` is expired: writers purge it, readers do not count it.

Keys, and every reader / writer of them (all in ``rate_limiter.py``):

- ``org_key(org_id)``: ZSET ``concurrent_calls:<org_id>``.
  Writers: ``acquire_slot`` (purge + add pending), ``renew_slot`` / ``claim_slot`` (re-add claimed),
  ``release_slot`` (remove). Readers: ``get_concurrent_count`` (purge + count), ``acquire_slot`` (count).
- ``scope_key(scope)``: ZSET ``concurrent_calls:<scope>`` (e.g. ``campaign:<id>``), same writers and the
  ``acquire_slot`` count. Scope values never collide with org keys: they always carry a ``<kind>:`` prefix.
- ``rate_limiter.FLEET_CONCURRENT_KEY``: ZSET ``concurrent_calls_fleet``, the fleet-wide mirror of every org slot
  (autoscaling signal). Writers: ``acquire_slot`` (purge + add), ``renew_slot`` / ``claim_slot``, ``release_slot``.
  Reader: ``get_fleet_concurrent_count`` (count of members with score > ``TIME``, no write).
- ``mapping_key(workflow_run_id)``: HASH ``workflow_slot_mapping:<run>`` with ``org_id``, ``slot_id`` (the
  attempt_id), optional ``scope_key``, and one ``sem:<zset key>`` field per provider semaphore member of the
  attempt (none are written yet: the semaphores arrive with the B1 capacity rows; ``release_slot`` already
  removes them). Writers: ``store_workflow_slot_mapping[_if_absent]`` (bind), ``renew_slot`` (refresh TTL,
  re-create after a Redis restart), ``release_slot`` (delete with its slot). Readers:
  ``get_workflow_slot_mapping``, ``claim_slot``, ``reconcile_workflow_slot_mapping``.
"""

import uuid


def new_attempt_id() -> str:
    """The admission id: the one member every slot key of the attempt carries."""
    return uuid.uuid4().hex


def org_key(organization_id: int) -> str:
    return f"concurrent_calls:{organization_id}"


def scope_key(scope: str | None) -> str:
    """Empty when the slot has no scope: the Lua scripts skip an empty key."""
    return f"concurrent_calls:{scope}" if scope else ""


def mapping_key(workflow_run_id: int | None) -> str:
    """Empty when the slot is not bound to a run yet: the Lua scripts skip an empty key."""
    return f"workflow_slot_mapping:{workflow_run_id}" if workflow_run_id else ""
