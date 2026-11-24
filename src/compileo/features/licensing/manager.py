import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple, Any, Dict
from .connector import LicenseConnector
from .models import LicenseStatus
from .encryption import encrypt_value, decrypt_value
from ...storage.src.database import get_system_db_connection

import logging

logger = logging.getLogger(__name__)

class LicenseManager:
    """Manages license lifecycle and validation using the dedicated licenses table."""
    
    def __init__(self):
        self.connector = LicenseConnector()

    def get_instance_id(self, owner_id: str) -> str:
        """Gets or creates a unique instance ID for this user's license."""
        with get_system_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT instance_id FROM licenses WHERE owner_id = ?", (owner_id,))
            row = cursor.fetchone()
            if row and row[0]:
                return row[0]
            
            # Use hardware-based stable ID instead of random UUID
            import uuid as uuid_mod
            node_id = uuid_mod.getnode()
            instance_id = f"machine-{node_id}"
            
            cursor.execute(
                "INSERT OR IGNORE INTO licenses (owner_id, instance_id) VALUES (?, ?)",
                (owner_id, instance_id)
            )
            cursor.execute(
                "UPDATE licenses SET instance_id = ? WHERE owner_id = ? AND instance_id IS NULL",
                (instance_id, owner_id)
            )
            conn.commit()
            return instance_id

    async def activate(self, license_key: str, owner_id: str) -> Tuple[bool, str]:
        """Activates a license key for a specific user."""
        instance_id = self.get_instance_id(owner_id)
        
        # 1. First, check if this machine is already activated for this key
        logger.info(f"Checking for existing activation of {license_key} for {instance_id}")
        license_info = await self.connector.get_license(license_key)
        
        if license_info.get('success') and 'data' in license_info:
            data = license_info['data']
            activations = data.get('activations', [])
            
            for activation in activations:
                if activation.get('label') == instance_id and not activation.get('deactivated_at'):
                    logger.info(f"Found existing active token for this machine: {activation.get('token')}")
                    
                    # We found a valid token! Save it and return success
                    expiry_str = data.get('expires_at')
                    
                    with get_system_db_connection() as conn:
                        cursor = conn.cursor()
                        now = datetime.utcnow().isoformat()
                        cursor.execute('''
                            INSERT OR REPLACE INTO licenses
                            (owner_id, license_key_encrypted, activation_token_encrypted, status, expiry, last_check, instance_id, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            owner_id,
                            encrypt_value(license_key),
                            encrypt_value(activation.get('token')),
                            'active',
                            expiry_str,
                            now,
                            instance_id,
                            now
                        ))
                        conn.commit()
                        return True, "License restored successfully"

        # 2. If not found, proceed with normal activation
        response = await self.connector.activate(license_key, instance_id)
        
        if not response.success or not response.data:
            # 3. If limit reached, try to intelligently recover by replacing an old activation
            if response.code == 'license_activation_limit_reached' and license_info.get('success'):
                data = license_info.get('data', {})
                activations = data.get('activations', [])
                active_activations = [a for a in activations if not a.get('deactivated_at')]
                
                # If there's only one active session and it's not us, deactivate it and try again
                if len(active_activations) == 1:
                    old_token = active_activations[0].get('token')
                    logger.info(f"Limit reached. Deactivating old session {old_token} to make room.")
                    await self.connector.deactivate(old_token)
                    
                    # Retry activation
                    response = await self.connector.activate(license_key, instance_id)
                    if response.success and response.data:
                        return await self._save_license_and_return(license_key, response.data, owner_id, instance_id)

            error_msg = response.message or "Activation failed"
            if response.code == 'license_activation_limit_reached':
                error_msg = f"License has reached its activation limit. If you are re-installing on the same machine, please ensure your machine ID ({instance_id}) matches your previous installation or contact support to reset your activations."
            
            return False, error_msg
        
        return await self._save_license_and_return(license_key, response.data, owner_id, instance_id)

    async def _save_license_and_return(self, license_key: str, data: Any, owner_id: str, instance_id: str) -> Tuple[bool, str]:
        """Helper to save license data to DB and return success."""
        # Handle cases where data is ActivationResponse data or dict
        token = data.token if hasattr(data, 'token') else data.get('token')
        license_obj = data.license if hasattr(data, 'license') else data.get('license')
        expiry = None
        if license_obj:
            expiry = license_obj.expires_at if hasattr(license_obj, 'expires_at') else license_obj.get('expires_at')
            
        with get_system_db_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            
            # Convert expiry to string if it's a datetime
            if expiry and hasattr(expiry, 'isoformat'):
                expiry = expiry.isoformat()
                
            cursor.execute('''
                INSERT OR REPLACE INTO licenses
                (owner_id, license_key_encrypted, activation_token_encrypted, status, expiry, last_check, instance_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                owner_id,
                encrypt_value(license_key),
                encrypt_value(token),
                'active',
                expiry,
                now,
                instance_id,
                now
            ))
            conn.commit()
            return True, "License activated successfully"

    async def validate(self, owner_id: str, force: bool = False) -> bool:
        """Validates the license (cached or remote) for a user."""
        status = self.get_local_status(owner_id)
        if not status or status.status != 'active':
            return False
        
        # Check if stale (more than 24h)
        if not force and (datetime.utcnow() - status.last_check) < timedelta(hours=24):
            return True
        
        # Remote validation
        if not status.token:
            return False
            
        response = await self.connector.validate(status.token)
        if response.success and response.data and not response.data.deactivated_at:
            # Update last check
            self._update_last_check(owner_id)
            return True
        else:
            # Mark as inactive
            self._set_status('inactive', owner_id)
            return False

    def get_local_status(self, owner_id: str, license_key: Optional[str] = None) -> Optional[LicenseStatus]:
        """Gets the current license status from the licenses table."""
        try:
            with get_system_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT license_key_encrypted, activation_token_encrypted, status, expiry, last_check
                    FROM licenses
                    WHERE owner_id = ?
                ''', (owner_id,))
                
                rows = cursor.fetchall()
                if not rows:
                    return None
                
                selected_row = None
                if license_key:
                    for r in rows:
                        try:
                            if decrypt_value(r[0]) == license_key:
                                selected_row = r
                                break
                        except: continue
                    if not selected_row: return None
                else:
                    selected_row = rows[0]

                if not selected_row[0]:
                    return None
                
                enc_key, enc_token, status, expiry_str, last_check_str = selected_row
                
                try:
                    key = decrypt_value(enc_key)
                    token = decrypt_value(enc_token) if enc_token else None
                except:
                    return None
                    
                last_check = datetime.fromisoformat(last_check_str) if last_check_str else datetime.min
                expiry = datetime.fromisoformat(expiry_str) if expiry_str else None
                
                return LicenseStatus(
                    key=key,
                    status=status or 'inactive',
                    expiry=expiry,
                    last_check=last_check,
                    token=token
                )
        except Exception:
            return None

    def _update_last_check(self, owner_id: str):
        with get_system_db_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            cursor.execute(
                "UPDATE licenses SET last_check = ?, updated_at = ? WHERE owner_id = ?",
                (now, now, owner_id)
            )
            conn.commit()

    def _set_status(self, status: str, owner_id: str):
        with get_system_db_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            cursor.execute(
                "UPDATE licenses SET status = ?, updated_at = ? WHERE owner_id = ?",
                (status, now, owner_id)
            )
            conn.commit()
