import os
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional

# Default salt for key derivation (used when no salt file exists)
DEFAULT_SALT = b'\x12\xaf\x89\x34\xde\xad\xbe\xef\x00\x11\x22\x33'


def _get_salt() -> bytes:
    """Get or generate salt for encryption key derivation."""
    salt_path = Path("storage/.salt")
    if salt_path.exists():
        return salt_path.read_bytes()
    else:
        # Generate new salt and save it
        salt = os.urandom(16)
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        salt_path.write_bytes(salt)
        # Restrict file permissions (owner read/write only)
        if os.name != 'nt':  # Unix-like systems
            salt_path.chmod(0o600)
        return salt


def _get_key() -> bytes:
    """Derive encryption key from environment variable or fallback to default."""
    # Get password from environment variable with fallback for development
    env_password = os.environ.get("COMPILEO_ENCRYPTION_KEY")
    if env_password:
        password = env_password.encode()
    else:
        # Fallback for development only - not recommended for production
        password = b"compileo-license-system-secret-key-2026"
    
    # Use per-instance salt if available, otherwise use default
    salt = _get_salt()
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password))

def encrypt_value(value: str) -> str:
    """Encrypts a string value."""
    if not value:
        return ""
    f = Fernet(_get_key())
    return f.encrypt(value.encode()).decode()

def decrypt_value(encrypted_value: str) -> str:
    """Decrypts an encrypted string value."""
    if not encrypted_value:
        return ""
    try:
        f = Fernet(_get_key())
        return f.decrypt(encrypted_value.encode()).decode()
    except Exception:
        return ""
