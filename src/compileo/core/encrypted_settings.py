"""
Registry and utilities for encrypted settings.

This module manages which settings keys require encryption at rest
and provides helper functions for encryption/decryption operations.
"""

from typing import Set

# Settings keys that must be encrypted at rest
ENCRYPTED_SETTING_KEYS: Set[str] = {
    "openai_api_key",
    "grok_api_key",
    "gemini_api_key",
    "huggingface_api_key",
    "huggingface_hub_token",
}

# Patterns that indicate a sensitive key (for future-proofing)
SENSITIVE_KEY_PATTERNS = [
    "api_key",
    "token",
    "secret",
    "password",
    "credential",
]


def is_sensitive_key(key: str) -> bool:
    """
    Check if a setting key contains sensitive data that should be encrypted.
    
    Args:
        key: The setting key to check
        
    Returns:
        True if the key should be encrypted, False otherwise
    """
    # Check exact matches first
    if key in ENCRYPTED_SETTING_KEYS:
        return True
    
    # Check for sensitive patterns in key name
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in SENSITIVE_KEY_PATTERNS)


def register_sensitive_key(key: str) -> None:
    """
    Register a new key as requiring encryption.
    
    Args:
        key: The setting key to register as sensitive
    """
    ENCRYPTED_SETTING_KEYS.add(key)
