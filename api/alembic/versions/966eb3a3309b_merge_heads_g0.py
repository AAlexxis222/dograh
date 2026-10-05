"""merge heads g0

Joins XPAND's a7c2e9d41b06 (workflow_runs.effective_configurations) with upstream's
3a7b91c5d402 (transfer agent tool). Both descend from f3a1c47b9e02, so no database can
reach a single head without this merge. A merge revision is safe whatever either side
has already applied; nothing is re-parented.

Revision ID: 966eb3a3309b
Revises: a7c2e9d41b06, 3a7b91c5d402
Create Date: 2026-10-05 07:52:23.497735

"""

from typing import Sequence, Union

revision: str = "966eb3a3309b"
down_revision: Union[str, None] = ("a7c2e9d41b06", "3a7b91c5d402")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
