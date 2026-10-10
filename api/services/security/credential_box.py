"""App-level AEAD envelope for credentials at rest (VOZ-AC-B6-15/16).

Token: ``v1:<key_id>:<nonce_b64>:<ct_b64>`` (AES-256-GCM, 96-bit random nonce, the tag at the end of ``ct``).
AAD = ``f"{table}:{pk}:{column}:{json_path}"`` so a token copied to another row does not decrypt; the surfaces a
token may live in are listed in ``secret_surfaces.SURFACES``.

``CREDENTIALS_MASTER_KEY`` is ``key_id=base64(32 bytes)``, comma-separated, the active key first; older keys only
decrypt. A cell validates it at start (``cell_startup``); anywhere else a missing key fails on first use.

Light on purpose (stdlib + cryptography): the cell start gate imports it in every role.
"""

import base64
import binascii
import functools
import os
import re

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ENV = "CREDENTIALS_MASTER_KEY"
_VERSION = "v1"
_KEY_BYTES = 32
_NONCE_BYTES = 12
# ":" separates token parts and "," separates keys, so a key id never carries either.
_KEY_ID = re.compile(r"[A-Za-z0-9_.-]+")


class MasterKeyInvalid(RuntimeError):
    """VOZ-AC-B0-28 shape. Never carries key material: an item is named by its position, not its text."""

    def __init__(self, where: str, reason: str, hint: str):
        super().__init__(
            f"code=credentials_master_key_invalid where={where} reason={reason} hint={hint}"
        )
        self.code, self.where, self.reason, self.hint = (
            "credentials_master_key_invalid",
            where,
            reason,
            hint,
        )


class CredentialUnreadable(Exception):
    """VOZ-AC-B0-28 shape. ``where`` is the AAD (table:pk:column:json_path); the token is never echoed."""

    def __init__(self, where: str, reason: str):
        self.code, self.where, self.reason = "credential_unreadable", where, reason
        self.hint = "restore CREDENTIALS_MASTER_KEY history (Secrets Manager) or re-enter the credential"
        super().__init__(
            f"code={self.code} where={where} reason={reason} hint={self.hint}"
        )


def master_key_items(raw: str) -> list[str]:
    """Non-empty ``key_id=base64`` items of ``raw``, unvalidated (log redaction reads them too)."""
    return [item for item in (part.strip() for part in raw.split(",")) if item]


def _parse_key(where: str, item: str, seen: set[str]) -> tuple[str, bytes]:
    # Errors name the item by position only: a mis-shaped item may be the raw key itself.
    key_id, _, b64 = (part.strip() for part in item.partition("="))
    if not b64:
        raise MasterKeyInvalid(
            where,
            "item is not 'key_id=base64'",
            f"write {ENV} as 'id=base64(32 bytes)'",
        )
    if not _KEY_ID.fullmatch(key_id):
        raise MasterKeyInvalid(
            where,
            "key id must be non-empty letters, digits, '_', '.' or '-'",
            "rename the key id (':' and ',' are separators)",
        )
    if key_id in seen:
        raise MasterKeyInvalid(
            where, "key id appears twice", "give every key a distinct id"
        )
    try:
        key = base64.b64decode(b64, validate=True)
    except binascii.Error:
        raise MasterKeyInvalid(
            where,
            "key is not valid base64",
            "encode the 32 key bytes with standard base64",
        ) from None
    if len(key) != _KEY_BYTES:
        raise MasterKeyInvalid(
            where,
            f"key is {len(key)} bytes, need {_KEY_BYTES}",
            f"generate a key with: openssl rand -base64 {_KEY_BYTES}",
        )
    return key_id, key


@functools.cache
def load_keys() -> tuple[tuple[str, bytes], ...]:
    """Validated ``(key_id, key)`` pairs, active first. Raises ``MasterKeyInvalid``; a failure is not cached."""
    items = master_key_items(os.environ.get(ENV, ""))
    if not items:
        raise MasterKeyInvalid(
            "env", f"{ENV} is missing", "inject it from the cell Secrets Manager"
        )
    keys: list[tuple[str, bytes]] = []
    for index, item in enumerate(items):
        keys.append(_parse_key(f"{ENV}[{index}]", item, {key_id for key_id, _ in keys}))
    return tuple(keys)


def encrypt(plaintext: bytes, aad: bytes) -> str:
    if not aad:
        raise ValueError(
            "code=credential_aad_missing where=credential_box.encrypt reason=empty aad binds the token "
            "to no row hint=pass aad=f'{table}:{pk}:{column}:{json_path}'"
        )
    key_id, key = load_keys()[0]
    nonce = os.urandom(_NONCE_BYTES)
    ct = AESGCM(key).encrypt(nonce, plaintext, aad)
    return f"{_VERSION}:{key_id}:{base64.b64encode(nonce).decode()}:{base64.b64encode(ct).decode()}"


def decrypt(token: str, aad: bytes) -> bytes:
    """Plaintext of ``token``. Any unreadable token raises ``CredentialUnreadable``: no fallback value, ever."""
    where = aad.decode("utf-8", "replace")
    if not isinstance(token, str):
        raise CredentialUnreadable(where, "stored value is not a string")
    parts = token.split(":")
    if len(parts) != 4 or parts[0] != _VERSION:
        raise CredentialUnreadable(where, "stored value is not a v1 token")
    _, key_id, nonce_b64, ct_b64 = parts
    key = dict(load_keys()).get(key_id)
    if key is None:
        raise CredentialUnreadable(
            where, "token key id is not in CREDENTIALS_MASTER_KEY"
        )
    try:
        nonce, ct = (
            base64.b64decode(nonce_b64, validate=True),
            base64.b64decode(ct_b64, validate=True),
        )
    except binascii.Error:
        raise CredentialUnreadable(where, "token part is not valid base64") from None
    if len(nonce) != _NONCE_BYTES:
        raise CredentialUnreadable(
            where, f"token nonce is {len(nonce)} bytes, need {_NONCE_BYTES}"
        )
    try:
        return AESGCM(key).decrypt(nonce, ct, aad)
    except InvalidTag:
        raise CredentialUnreadable(
            where, "authentication failed (wrong row or key)"
        ) from None
