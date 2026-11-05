"""VOZ-N0-24a: plaintext, indexed account_id on telephony_configurations

Inbound dispatch and the duplicate-account guard looked a call up by reading inside the
`credentials` JSON (`credentials->>'<account field>'`). A later revision encrypts that column
whole, so the account id moves to a column of its own: it identifies an account, it is not a
secret (it is already logged on every inbound match).

Expand only: a nullable column, a composite index, and a backfill of the existing rows. The
backfill is SQL, per provider, from the credential field that identifies the account. It reads
only rows whose credentials are a JSON object, and only writes a value it found, so a re-run can
never blank a routing key. A string or a JSON number is the account id (a number as the text
`->>` renders, which is what the old lookup compared); a missing or empty field, an array, a
boolean, a legacy document stored double-encoded or a JSON null gets NULL: the old lookup did not
match those either.

Rows saved by the previous release while this one rolls out, or after an image rollback, keep a
NULL account_id; the inbound lookup and the duplicate-account check heal them on first use
(TelephonyPhoneNumberClient), so no operator step is needed.

downgrade() drops the index and the column. Roll the image back FIRST: this release's pods select
the column and fail on every telephony query once it is gone. Both directions set a short
lock_timeout so a deploy blocked by a long transaction on the table fails fast instead of
queueing behind it (and every inbound lookup behind the ALTER).

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
# JSON type, so json_typeof tells a string or number from the rest, and `->>` renders the text.
_EXPECTED = (
    "CASE WHEN json_typeof(credentials -> CAST(:field AS text)) IN ('string', 'number') "
    "THEN NULLIF(credentials ->> CAST(:field AS text), '') END"
)

LOCK_TIMEOUT = "5s"


def _fail_fast_on_locks() -> None:
    op.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")


def backfill(bind) -> None:
    """Fill ``account_id`` from ``credentials`` for every provider that has an account field.
    Only object documents are read and only a found value is written, so it can be repeated
    (the tests do) without ever blanking a row."""
    for provider, field in _ACCOUNT_ID_FIELDS.items():
        bind.execute(
            sa.text(
                f"UPDATE telephony_configurations SET account_id = {_EXPECTED} "
                "WHERE provider = :provider AND json_typeof(credentials) = 'object' "
                f"AND {_EXPECTED} IS NOT NULL"
            ),
            {"provider": provider, "field": field},
        )
        logger.info("telephony_configurations.account_id backfilled for %s", provider)


def upgrade() -> None:
    _fail_fast_on_locks()
    op.add_column(
        "telephony_configurations",
        sa.Column("account_id", sa.Text(), nullable=True),
    )
    op.create_index(INDEX_NAME, "telephony_configurations", ["provider", "account_id"])
    backfill(op.get_bind())


def downgrade() -> None:
    _fail_fast_on_locks()
    op.drop_index(INDEX_NAME, table_name="telephony_configurations")
    op.drop_column("telephony_configurations", "account_id")
