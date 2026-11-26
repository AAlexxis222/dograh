"""public links expire and can be revoked: expiry and revocation columns on workflow_runs

Expansion only (VOZ-N0-30, VOZ-AC-B6-51): two nullable columns, no backfill, nothing dropped or rewritten. A run that
already holds a public token keeps ``public_access_expires_at`` NULL, which the app reads as expired (fail closed: no
instance carries live links), so an old link stops opening the moment this code runs.

Revision ID: c4a8d2e61f07
Revises: e2b7c4d9a613
Create Date: 2026-11-05 18:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4a8d2e61f07"
down_revision: Union[str, None] = "e2b7c4d9a613"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "workflow_runs",
        sa.Column(
            "public_access_expires_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    op.add_column(
        "workflow_runs",
        sa.Column(
            "public_access_revoked_at", sa.DateTime(timezone=True), nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_column("workflow_runs", "public_access_revoked_at")
    op.drop_column("workflow_runs", "public_access_expires_at")
