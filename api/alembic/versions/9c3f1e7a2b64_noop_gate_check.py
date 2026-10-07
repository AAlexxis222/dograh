"""noop gate check

Revision ID: 9c3f1e7a2b64
Revises: d7e3a915c2b8
Create Date: 2026-10-07 09:00:00.000000

"""

from typing import Sequence, Union

revision: str = "9c3f1e7a2b64"
down_revision: Union[str, None] = "d7e3a915c2b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
