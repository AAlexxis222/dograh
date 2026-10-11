"""Credential references in the configuration a run freezes (VOZ-AC-B1-07, VOZ-AC-B6-15, VOZ-AT-B1-27).

A run never stores a credential value. At creation every non-empty secret leaf of its effective configuration (the
registered paths, ``WORKFLOW_CONFIGURATION_SECRET_PATHS``) becomes a reference to the same leaf of the definition the
run pins, or ``REDACTED`` when that definition does not hold the same value there. The organization layer refuses
secrets (422), so the pinned definition is the only legitimate source of one.

The two places that hand a run's configuration to the runtime resolve the references against the pinned definition:
``cascade.run_configurations_for`` and ``WorkflowRunClient.get_workflow_run_configurations``. A reference that no
longer resolves fails closed with ``CredentialRefUnresolved`` (VOZ-AC-B0-28 shape) rather than handing a provider a
placeholder. A plain value (a run frozen before the redaction migration) passes through until the contraction.

The reference names its target table and row, so the credential object (VOZ-AC-B1-77) can become the target without
changing the shape. Alembic revision c5e8a2d7f914 keeps a frozen copy of ``to_credential_refs`` for historic runs.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterator
from typing import Any

from api.services.security.secret_surfaces import WORKFLOW_CONFIGURATION_SECRET_PATHS
from api.services.security.secret_walk import iter_leaves, value_at

REDACTED = "<redacted>"
_REF = re.compile(r"<credential_ref:workflow_definitions:(\d+)>")
_EMPTY: tuple[Any, ...] = (None, "", [], {})


def credential_ref(definition_id: int) -> str:
    return f"<credential_ref:workflow_definitions:{definition_id}>"


def is_credential_marker(value: Any) -> bool:
    """A reference or ``REDACTED``: what a run stores in place of a credential, never a secret."""
    return isinstance(value, str) and (
        value == REDACTED or _REF.fullmatch(value) is not None
    )


class CredentialRefUnresolved(Exception):
    """VOZ-AC-B0-28 shape. Never carries a credential value."""

    def __init__(self, where: str, reason: str):
        self.code, self.where, self.reason = "credential_ref_unresolved", where, reason
        self.hint = (
            "the workflow version this run pins no longer holds the credential: "
            "re-enter it in the workflow and start a new run"
        )
        super().__init__(
            f"code={self.code} where={where} reason={reason} hint={self.hint}"
        )


def _live_secret_leaves(
    document: dict[str, Any],
) -> Iterator[tuple[tuple[str | int, ...], dict[str, Any], str, Any]]:
    """``(path, container, key, value)`` of every registered secret leaf, outermost first, read again at each step:
    a leaf the caller replaced (``credentials``) takes the leaves nested in it along, and those are skipped instead
    of being rewritten in a detached copy."""
    leaves = sorted(
        (
            leaf
            for pattern in WORKFLOW_CONFIGURATION_SECRET_PATHS
            for leaf in iter_leaves(document, pattern, (), indexed=True)
        ),
        key=lambda leaf: len(leaf[0]),
    )
    for path, container, key in leaves:
        found, value = value_at(document, path)
        if found:
            yield path, container, key, value


def to_credential_refs(
    effective: dict[str, Any], pinned: Any, definition_id: int | None
) -> dict[str, Any]:
    """Copy of ``effective`` with every non-empty secret leaf replaced: by a reference when ``pinned`` (the
    definition's own document) holds the same value at the same path, by ``REDACTED`` otherwise."""
    frozen = copy.deepcopy(effective)
    source = pinned if isinstance(pinned, dict) else {}
    for path, container, key, value in _live_secret_leaves(frozen):
        if value in _EMPTY or is_credential_marker(value):
            continue
        found, pinned_value = value_at(source, path)
        if definition_id is not None and found and pinned_value == value:
            container[key] = credential_ref(definition_id)
        else:
            container[key] = REDACTED
    return frozen


def stored_secret_paths(document: Any) -> list[str]:
    """Dotted paths of secret leaves holding a value rather than a marker; a run row must have none."""
    if not isinstance(document, dict):
        return []
    return [
        ".".join(map(str, path))
        for path, _container, _key, value in _live_secret_leaves(document)
        if value not in _EMPTY and not is_credential_marker(value)
    ]


def resolve_credential_refs(
    frozen: Any,
    pinned: Callable[[], Any],
    *,
    definition_id: int | None,
    run_id: int | None,
) -> Any:
    """Copy of ``frozen`` with every reference replaced by the pinned definition's value at the same path.

    ``pinned`` returns that definition's document and is called only when a reference is present, so a run without
    one never touches its definition. ``REDACTED`` and plain values are returned as they are."""
    resolved = copy.deepcopy(frozen)
    if not isinstance(resolved, dict):
        return resolved
    source: dict[str, Any] | None = None
    for path, container, key, value in _live_secret_leaves(resolved):
        match = _REF.fullmatch(value) if isinstance(value, str) else None
        if match is None:
            continue
        where = f"workflow_runs:{run_id}:effective_configurations:{'.'.join(map(str, path))}"
        if int(match.group(1)) != definition_id:
            raise CredentialRefUnresolved(
                where,
                f"the reference names workflow definition {match.group(1)}, "
                f"the run pins {definition_id}",
            )
        if source is None:
            document = pinned()
            source = document if isinstance(document, dict) else {}
        found, value = value_at(source, path)
        if not found or value in _EMPTY:
            raise CredentialRefUnresolved(
                where, f"workflow definition {definition_id} holds no value there"
            )
        container[key] = copy.deepcopy(value)
    return resolved
