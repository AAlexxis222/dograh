"""add effective_configurations to workflow_runs

Revision ID: a7c2e9d41b06
Revises: f3a1c47b9e02
Create Date: 2026-09-05 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7c2e9d41b06"
down_revision: Union[str, None] = "f3a1c47b9e02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Nullable on purpose: runs created before this column keep NULL and
    # readers fall back to the pinned definition (no retroactive org layer).
    op.add_column(
        "workflow_runs",
        sa.Column("effective_configurations", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("workflow_runs", "effective_configurations")
