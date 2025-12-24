"""
Settings persistence for the GUI using SQLite.
"""

import json
import sqlite3
from typing import Dict, Any, Optional
from src.compileo.storage.src.database import get_db_connection
from src.compileo.core.logging import get_logger

logger = get_logger(__name__)


class SettingsStorage:
    """Handles persistence of GUI settings using SQLite."""

    def __init__(self):
        """Initialize settings storage."""
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the gui_settings table exists."""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gui_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT,  -- JSON serialized value
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        finally:
            conn.close()

    def load_settings(self) -> Dict[str, Any]:
        """Load all settings from database.

        Returns:
            Dict containing all settings
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM gui_settings')
            rows = cursor.fetchall()

            settings = {}
            for row in rows:
                key, value = row
                try:
                    # Try to parse as JSON
                    settings[key] = json.loads(value) if value else None
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, store as string
                    settings[key] = value
            return settings
        finally:
            conn.close()

    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save all settings to database.

        Args:
            settings: Settings dictionary to save

        Returns:
            True if successful, False otherwise
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            # Clear existing settings
            cursor.execute('DELETE FROM gui_settings')

            # Insert new settings
            for key, value in settings.items():
                # Serialize value to JSON if it's not a string
                if not isinstance(value, str):
                    try:
                        serialized_value = json.dumps(value)
                    except (TypeError, ValueError):
                        serialized_value = str(value)
                else:
                    serialized_value = value

                cursor.execute('''
                    INSERT INTO gui_settings (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (key, serialized_value))

            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error: Could not save settings: {e}")
            return False
        finally:
            conn.close()

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value.

        Args:
            key: Setting key
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM gui_settings WHERE key = ?', (key,))
            row = cursor.fetchone()

            if row:
                value = row[0]
                try:
                    # Try to parse as JSON
                    return json.loads(value) if value else None
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, return as string
                    return value
            return default
        finally:
            conn.close()

    def set_setting(self, key: str, value: Any) -> bool:
        """Set a specific setting value.

        Args:
            key: Setting key
            value: Setting value

        Returns:
            True if successful, False otherwise
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            # Serialize value to JSON if it's not a string
            if not isinstance(value, str):
                try:
                    serialized_value = json.dumps(value)
                except (TypeError, ValueError):
                    serialized_value = str(value)
            else:
                serialized_value = value

            # Insert or replace
            cursor.execute('''
                INSERT OR REPLACE INTO gui_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, serialized_value))

            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error: Could not set setting: {e}")
            return False
        finally:
            conn.close()

    def reset_settings(self) -> bool:
        """Reset all settings to empty.

        Returns:
            True if successful, False otherwise
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM gui_settings')
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error: Could not reset settings: {e}")
            return False
        finally:
            conn.close()


# Global settings storage instance
settings_storage = SettingsStorage()