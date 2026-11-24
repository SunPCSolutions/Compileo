"""
Database connection utilities for Compileo storage with multi-tenant support.

This module provides thread-safe database connections with tenant isolation.
Each tenant (user) has their own database file for data isolation.
"""

import sqlite3
import os
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional

# Global SQLite lock for thread safety in compiled environments
_sqlite_lock = threading.Lock()


def get_db_lock() -> threading.Lock:
    """Get the global database lock for thread-safe operations."""
    return _sqlite_lock


@contextmanager
def db_lock_context() -> Generator[None, None, None]:
    """Context manager for database lock acquisition."""
    _sqlite_lock.acquire()
    try:
        yield
    finally:
        _sqlite_lock.release()


def get_base_storage_path() -> Path:
    """Get the base storage directory path."""
    return Path(__file__).parent.parent.parent.parent.parent / "storage"


def get_db_path(owner_id: Optional[str] = None, db_type: str = "tenant") -> str:
    """
    Get the database file path with multi-tenant support.
    
    Args:
        owner_id: User ID for tenant-specific databases. If None, returns system database.
        db_type: "system" for global auth data, "tenant" for user data
        
    Returns:
        str: Path to the database file
    """
    base_path = get_base_storage_path()
    
    if db_type == "system" or owner_id is None:
        # System database for users, sessions, quotas, licenses
        db_path = base_path / "system" / "system.db"
    else:
        # Tenant-specific database
        db_path = base_path / owner_id / "tenant.db"
    
    return str(db_path)


def initialize_tenant_database(owner_id: str) -> None:
    """
    Initialize a tenant-specific database for a user.
    
    Args:
        owner_id: The user ID to create the database for
    """
    db_path = get_db_path(owner_id=owner_id, db_type="tenant")
    db_file = Path(db_path)
    
    # Create directory with secure permissions
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Import here to avoid circular imports
    from .project.initialize_database import setup_tenant_schema
    
    # Initialize schema
    conn = sqlite3.connect(db_path)
    try:
        setup_tenant_schema(conn, owner_id)
    finally:
        conn.close()
    
    # Restrict file permissions (owner read/write only)
    if os.name != 'nt':  # Unix-like systems
        db_file.chmod(0o600)


@contextmanager
def get_db_connection(owner_id: Optional[str] = None, db_type: str = "tenant") -> Generator[sqlite3.Connection, None, None]:
    """
    Get a SQLite database connection with multi-tenant support.
    
    Args:
        owner_id: User ID for tenant-specific databases
        db_type: "system" for global auth data, "tenant" for user data
        
    Yields:
        sqlite3.Connection: Database connection object
    """
    import logging
    logger = logging.getLogger(__name__)
    
    db_path = get_db_path(owner_id, db_type)
    storage_dir = Path(db_path).parent
    
    # Ensure storage directory exists
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage directory created/verified: {storage_dir}")
    except Exception as e:
        logger.error(f"Failed to create storage directory: {e}")
        raise
    
    # Check permissions
    if not os.access(storage_dir, os.W_OK):
        logger.error(f"Storage directory is not writable: {storage_dir}")
        raise PermissionError(f"Cannot write to storage directory: {storage_dir}")
    
    # Create connection
    conn = None
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        logger.info(f"Database connection successful: {db_path}")
        
        # Enable WAL mode for better concurrent access (important for Docker multi-worker setups)
        conn.execute("PRAGMA journal_mode=WAL")
        logger.debug(f"WAL mode enabled for: {db_path}")
        
        # Enable foreign key enforcement for referential integrity
        conn.execute("PRAGMA foreign_keys = ON")
        logger.debug(f"Foreign keys enabled for: {db_path}")
        
        # Enable row factory for dictionary-like access
        conn.row_factory = sqlite3.Row
        
        yield conn
        
    except Exception as e:
        logger.error(f"Failed to connect to database {db_path}: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info(f"Database connection closed: {db_path}")


@contextmanager
def get_system_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Get a connection to the system database (for auth, users, sessions).
    
    Yields:
        sqlite3.Connection: System database connection
    """
    with get_db_connection(owner_id=None, db_type="system") as conn:
        yield conn


@contextmanager
def get_tenant_db_connection(owner_id: str) -> Generator[sqlite3.Connection, None, None]:
    """
    Get a connection to a tenant-specific database.
    
    Args:
        owner_id: The user ID for the tenant
        
    Yields:
        sqlite3.Connection: Tenant database connection
    """
    # Initialize database if it doesn't exist
    db_path = Path(get_db_path(owner_id=owner_id, db_type="tenant"))
    if not db_path.exists():
        initialize_tenant_database(owner_id)
    
    with get_db_connection(owner_id=owner_id, db_type="tenant") as conn:
        yield conn


# Legacy function for backwards compatibility during transition
@contextmanager
def get_legacy_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Legacy database connection - returns system database connection.
    
    DEPRECATED: Use get_system_db_connection() or get_tenant_db_connection() instead.
    """
    with get_system_db_connection() as conn:
        yield conn
