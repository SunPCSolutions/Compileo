"""
Settings persistence for the GUI using SQLite with encryption support.
"""

import json
import sqlite3
import logging
from typing import Dict, Any, Optional

from src.compileo.storage.src.database import get_tenant_db_connection, get_system_db_connection
from src.compileo.core.encrypted_settings import is_sensitive_key
from src.compileo.features.licensing.encryption import encrypt_value, decrypt_value

logger = logging.getLogger(__name__)


class SettingsStorage:
    """Handles persistence of GUI settings using SQLite with encryption for sensitive data."""

    def __init__(self):
        """Initialize settings storage."""
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the gui_settings table exists in both system and tenant databases."""
        # System database
        # Note: Using empty string '' instead of NULL for system settings to ensure
        # UNIQUE constraint works properly (SQLite treats NULL != NULL in unique constraints)
        with get_system_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gui_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT,
                    is_encrypted BOOLEAN DEFAULT 0,
                    owner_id TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(key, owner_id)
                )
            ''')
            conn.commit()

    def _ensure_tenant_table(self, owner_id: str):
        """Ensure the gui_settings table exists in tenant database."""
        with get_tenant_db_connection(owner_id) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gui_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT,
                    is_encrypted BOOLEAN DEFAULT 0,
                    owner_id TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(key, owner_id)
                )
            ''')
            conn.commit()

    def _get_connection(self, owner_id: Optional[str] = None):
        """Get appropriate database connection."""
        if owner_id:
            return get_tenant_db_connection(owner_id)
        return get_system_db_connection()

    def load_settings(self, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load all settings from database.
        Loads system defaults first, then overrides with tenant-specific settings.
        
        Args:
            owner_id: Optional user ID to filter settings

        Returns:
            Dict containing all settings with sensitive values decrypted
        """
        settings = {}
        
        # First load system defaults (owner_id = '' for system-wide settings)
        try:
            with get_system_db_connection() as conn:
                cursor = conn.cursor()
                # DEBUG: Log the query execution for diagnosis
                logger.debug(f"DEBUG: Loading system settings with owner_id=''")
                cursor.execute('SELECT key, value, is_encrypted FROM gui_settings WHERE owner_id = ?', ('',))
                rows = cursor.fetchall()
                
                for row in rows:
                    key, value, is_encrypted = row
                    try:
                        if is_encrypted and value:
                            settings[key] = decrypt_value(value)
                        else:
                            settings[key] = json.loads(value) if value else None
                    except (json.JSONDecodeError, TypeError):
                        settings[key] = value
        except Exception:
            pass
        
        # Then load/override with user-specific settings if owner_id provided
        if owner_id:
            try:
                # Ensure table exists first
                self._ensure_tenant_table(owner_id)
                
                with get_tenant_db_connection(owner_id) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT key, value, is_encrypted FROM gui_settings WHERE owner_id = ? OR owner_id = ?',
                        (owner_id, '')
                    )
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        key, value, is_encrypted = row
                        try:
                            if is_encrypted and value:
                                settings[key] = decrypt_value(value)
                            else:
                                settings[key] = json.loads(value) if value else None
                        except (json.JSONDecodeError, TypeError):
                            settings[key] = value
            except Exception:
                pass
        
        return settings

    def save_settings(self, settings: Dict[str, Any], owner_id: Optional[str] = None) -> bool:
        """
        Save all settings to database with automatic encryption for sensitive keys.
        
        Args:
            settings: Settings dictionary to save
            owner_id: Optional user ID to associate settings with

        Returns:
            True if successful, False otherwise
        """
        if not settings:
            return True
            
        # Ensure tenant table exists if owner_id provided
        if owner_id:
            self._ensure_tenant_table(owner_id)
            
        try:
            with self._get_connection(owner_id) as conn:
                cursor = conn.cursor()

                # Insert new settings
                for key, value in settings.items():
                    # Determine if this key should be encrypted
                    is_encrypted = is_sensitive_key(key)
                    
                    # Encrypt if needed
                    if is_encrypted and value:
                        if isinstance(value, str):
                            serialized_value = encrypt_value(value)
                        else:
                            serialized_value = encrypt_value(str(value))
                    else:
                        # Serialize value to JSON if it's not a string
                        if not isinstance(value, str):
                            try:
                                serialized_value = json.dumps(value)
                            except (TypeError, ValueError):
                                serialized_value = str(value)
                        else:
                            serialized_value = value

                    # Use empty string for system settings (owner_id is None means system-wide setting)
                    db_owner_id = '' if owner_id is None else owner_id

                    cursor.execute('''
                        INSERT OR REPLACE INTO gui_settings (key, value, is_encrypted, owner_id, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (key, serialized_value, is_encrypted, db_owner_id))

                conn.commit()
                return True
        except Exception:
            return False

    def get_setting(self, key: str, default: Any = None, owner_id: Optional[str] = None) -> Any:
        """
        Get a specific setting value with automatic decryption for sensitive keys.
        
        Args:
            key: Setting key
            default: Default value if key not found
            owner_id: Optional user ID to filter settings

        Returns:
            Setting value or default (decrypted if sensitive)
        """
        with self._get_connection(owner_id) as conn:
            cursor = conn.cursor()
            if owner_id:
                # Try user-specific first, then global (owner_id = '' for system settings)
                cursor.execute(
                    'SELECT value, is_encrypted FROM gui_settings WHERE key = ? AND owner_id = ?',
                    (key, owner_id)
                )
                row = cursor.fetchone()
                if not row:
                    cursor.execute(
                        'SELECT value, is_encrypted FROM gui_settings WHERE key = ? AND owner_id = ?',
                        (key, '')
                    )
                    row = cursor.fetchone()
            else:
                cursor.execute(
                    'SELECT value, is_encrypted FROM gui_settings WHERE key = ? AND owner_id = ?',
                    (key, '')
                )
                row = cursor.fetchone()

            if row:
                value, is_encrypted = row
                try:
                    # Decrypt if needed
                    if is_encrypted and value:
                        decrypted = decrypt_value(value)
                        return decrypted if decrypted else default
                    
                    # Try to parse as JSON
                    return json.loads(value) if value else None
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, return as string
                    return value
            return default

    def set_setting(self, key: str, value: Any, owner_id: Optional[str] = None) -> bool:
        """
        Set a specific setting value with automatic encryption for sensitive keys.
        
        Args:
            key: Setting key
            value: Setting value
            owner_id: Optional user ID to associate setting with

        Returns:
            True if successful, False otherwise
        """
        # Determine if this key should be encrypted
        is_encrypted = is_sensitive_key(key)
        
        # Encrypt if needed
        if is_encrypted and value:
            if isinstance(value, str):
                serialized_value = encrypt_value(value)
            else:
                serialized_value = encrypt_value(str(value))
        else:
            # Serialize value to JSON if it's not a string
            if not isinstance(value, str):
                try:
                    serialized_value = json.dumps(value)
                except (TypeError, ValueError):
                    serialized_value = str(value)
            else:
                serialized_value = value

        with self._get_connection(owner_id) as conn:
            cursor = conn.cursor()

            # Use empty string for system settings (owner_id is None means system-wide setting)
            db_owner_id = '' if owner_id is None else owner_id

            # Insert or replace
            cursor.execute('''
                INSERT OR REPLACE INTO gui_settings (key, value, is_encrypted, owner_id, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (key, serialized_value, is_encrypted, db_owner_id))

            conn.commit()
            return True

    def reset_settings(self, owner_id: Optional[str] = None) -> bool:
        """
        Reset all settings for a user.
        
        Args:
            owner_id: Optional user ID to clear settings for

        Returns:
            True if successful, False otherwise
        """
        with self._get_connection(owner_id) as conn:
            cursor = conn.cursor()
            if owner_id:
                cursor.execute('DELETE FROM gui_settings WHERE owner_id = ?', (owner_id,))
            else:
                # System-wide settings use empty string '', not NULL
                cursor.execute('DELETE FROM gui_settings WHERE owner_id = ?', ('',))
            conn.commit()
            return True


# Global settings storage instance
settings_storage = SettingsStorage()
