from api.db import (
    credentials_at_rest,  # noqa: F401 registers the encryption-at-rest ORM events
)
from api.db.db_client import DBClient

db_client = DBClient()
