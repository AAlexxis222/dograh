"""Classification of storage-driver errors, so code outside api/db never imports the driver to make it."""

import asyncpg.exceptions
import sqlalchemy.exc

# What this stack (SQLAlchemy async + asyncpg) raises when Postgres is down, restarting, saturated or drops the
# connection. OSError covers ConnectionError, ConnectionRefusedError and the builtin TimeoutError;
# sqlalchemy.exc.TimeoutError is an exhausted pool.
_STORE_UNREACHABLE_ERRORS = (
    OSError,
    sqlalchemy.exc.TimeoutError,
    sqlalchemy.exc.InterfaceError,
    asyncpg.exceptions.PostgresConnectionError,
    asyncpg.exceptions.CannotConnectNowError,
    asyncpg.exceptions.TooManyConnectionsError,
    asyncpg.exceptions.AdminShutdownError,
)


def is_store_unreachable(exc: BaseException) -> bool:
    """Whether a lookup failed because the database could not be reached, not because of the request: one of
    _STORE_UNREACHABLE_ERRORS, or a DBAPIError whose connection really closed."""
    return isinstance(exc, _STORE_UNREACHABLE_ERRORS) or (
        isinstance(exc, sqlalchemy.exc.DBAPIError) and exc.connection_invalidated
    )
