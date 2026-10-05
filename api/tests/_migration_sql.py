"""Run one alembic revision's ``upgrade()`` against the test DB, inside the test's transaction.

The test database is already at head; data migrations are idempotent backfills, so seeding
legacy-shaped rows and running ``upgrade()`` again exercises their real SQL.
"""

import importlib.util
from pathlib import Path

from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy.ext.asyncio import AsyncSession

_VERSIONS = Path(__file__).resolve().parents[1] / "alembic" / "versions"


def migration_path(revision: str) -> Path:
    (path,) = _VERSIONS.glob(f"{revision}_*.py")
    return path


def load_migration(revision: str):
    spec = importlib.util.spec_from_file_location(
        f"migration_{revision}", migration_path(revision)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_upgrade(session: AsyncSession, migration) -> None:
    def _upgrade(sync_conn):
        with Operations.context(MigrationContext.configure(sync_conn)):
            migration.upgrade()

    connection = await session.connection()
    await connection.run_sync(_upgrade)
