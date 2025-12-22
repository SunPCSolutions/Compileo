"""
Database connection utilities for Compileo storage.
"""

import sqlite3
import os
from pathlib import Path


def get_db_path():
    """
    Get the database file path.

    Returns:
        str: Path to the database file
    """
    import logging
    logger = logging.getLogger(__name__)

    # Database path - in storage directory from project root
    db_path = Path(__file__).parent.parent.parent.parent.parent / "storage" / "database.db"
    logger.info(f"Calculated database path: {db_path}")
    logger.info(f"Storage directory exists: {db_path.parent.exists()}")
    if db_path.parent.exists():
        logger.info(f"Storage directory writable: {os.access(db_path.parent, os.W_OK)}")
    return str(db_path)


def get_db_connection():
    """
    Get a SQLite database connection.

    Returns:
        sqlite3.Connection: Database connection object
    """
    import logging
    logger = logging.getLogger(__name__)

    db_path = get_db_path()
    storage_dir = Path(db_path).parent

    # Ensure storage directory exists
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Storage directory created/verified")
    except Exception as e:
        logger.error(f"Failed to create storage directory: {e}")

    # Check permissions
    if not os.access(storage_dir, os.W_OK):
        logger.error("Storage directory is not writable")

    # Create connection
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        logger.info("Database connection successful")
        
        # Enable row factory
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

