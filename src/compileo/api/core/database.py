from typing import Generator
import sqlite3
from fastapi import Depends
from ...storage.src.database import get_tenant_db_connection
from .dependencies import get_current_active_user
from ..models.user import User

# Standard FastAPI dependency pattern for database connections
# This handles the context manager yield properly
def get_db(current_user: User = Depends(get_current_active_user)) -> Generator[sqlite3.Connection, None, None]:
    """
    FastAPI dependency that yields a database connection.
    This ensures the connection is closed after the request is finished.
    """
    with get_tenant_db_connection(current_user.id) as conn:
        yield conn
