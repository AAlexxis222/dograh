from api.db import credentials_at_rest
from api.db.db_client import DBClient

# Encryption at rest of every registered secret column: ORM events on the mapped classes.
credentials_at_rest.register()

db_client = DBClient()
