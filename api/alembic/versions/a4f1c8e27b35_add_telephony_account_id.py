"""VOZ-N0-24a: plaintext, indexed account_id on telephony_configurations

Inbound dispatch and the duplicate-account guard looked a call up by reading inside the
`credentials` JSON (`credentials->>'<account field>'`). A later revision encrypts that column
whole, so the account id moves to a column of its own: it identifies an account, it is not a
secret (it is already logged on every inbound match).

Expand only: a nullable column, a composite index, and a backfill of the existing rows. The
backfill is SQL, per provider, from the credential field that identifies the account. A
credential that is not a JSON object (a legacy document stored double-encoded, a JSON null) or
whose field is missing, empty or not a string gets NULL: the old `->>` lookup did not match it
either, and the app writer applies the same rule to every new save.

An app instance that still runs the previous release keeps saving credentials without an
account_id during a rolling update; running `backfill` again after the rollout (it is
idempotent) closes that window.

downgrade() drops the index and the column.

Revision ID: a4f1c8e27b35
Revises: d7e3a915c2b8
Create Date: 2026-10-11 09:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a4f1c8e27b35"
down_revision: Union[str, None] = "d7e3a915c2b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

INDEX_NAME = "ix_telephony_configurations_provider_account_id"

# Frozen copy, as of this revision, of each provider's `account_id_credential_field`
# (api/services/telephony/providers/<provider>/__init__.py); migrations do not import
# application code. ARI has no account concept and is absent: its account_id stays NULL.
_ACCOUNT_ID_FIELDS: dict[str, str] = {
    "cloudonix": "domain_id",
    "exotel": "account_sid",
    "plivo": "auth_id",
    "telnyx": "connection_id",
    "twilio": "account_sid",
    "vobiz": "auth_id",
    "vonage": "api_key",
}

# The value a row must carry after the backfill. `credentials` is a `json` column: `->` keeps the
# JSON type, so json_typeof tells a string from a number, and `->>` renders the string.
_EXPECTED = (
    "CASE WHEN json_typeof(credentials) = 'object' "
    "AND json_typeof(credentials -> CAST(:field AS text)) = 'string' "
    "THEN NULLIF(credentials ->> CAST(:field AS text), '') END"
)


def backfill(bind) -> None:
    """Fill ``account_id`` from ``credentials`` for every provider that has an account field.
    Idempotent; the tests call it on seeded rows."""
    for provider, field in _ACCOUNT_ID_FIELDS.items():
        params = {"provider": provider, "field": field}
        bind.execute(
            sa.text(
                f"UPDATE telephony_configurations SET account_id = {_EXPECTED} "
                "WHERE provider = :provider"
            ),
            params,
        )
        logger.info("telephony_configurations.account_id backfilled for %s", provider)


def upgrade() -> None:
    op.add_column(
        "telephony_configurations",
        sa.Column("account_id", sa.Text(), nullable=True),
    )
    op.create_index(INDEX_NAME, "telephony_configurations", ["provider", "account_id"])
    backfill(op.get_bind())


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="telephony_configurations")
    op.drop_column("telephony_configurations", "account_id")
