"""
Backend settings accessor for Compileo.
Reads settings from the GUI settings database with encryption support.
"""

import logging
from typing import Any, Optional
import json
from enum import Enum

from .encrypted_settings import is_sensitive_key
from ..features.licensing.encryption import encrypt_value, decrypt_value


class LogLevel(str, Enum):
    """Log level options."""
    NONE = "none"
    ERROR = "error"
    DEBUG = "debug"


class BackendSettings:
    """Access GUI settings from the backend with encryption support for sensitive data."""

    def __init__(self, owner_id: Optional[str] = None):
        """
        Initialize BackendSettings.
        
        Args:
            owner_id: Optional owner_id to fetch user-specific settings.
        """
        self.owner_id = owner_id

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from settings."""
        return self.get_setting("openai_api_key", "")

    def get_grok_api_key(self) -> str:
        """Get Grok API key from settings."""
        return self.get_setting("grok_api_key", "")

    def get_gemini_api_key(self) -> str:
        """Get Gemini API key from settings."""
        return self.get_setting("gemini_api_key", "")

    def get_huggingface_hub_token(self) -> str:
        """Get Hugging Face Hub token from settings."""
        # Use the key that the GUI actually saves: 'huggingface_api_key'
        return self.get_setting("huggingface_api_key", "")

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value from the GUI settings database.
        Automatically decrypts sensitive values.

        Args:
            key: Setting key
            default: Default value if key not found or database unavailable

        Returns:
            Setting value or default
        """
        # Import here to avoid circular imports
        from ..storage.src.database import get_tenant_db_connection, get_system_db_connection
        
        # Determine which database to use
        # User-specific settings are stored in tenant database
        try:
            if self.owner_id:
                # Check system database FIRST for global settings
                # This ensures system-wide settings (like log_level) are not shadowed
                # by stale values in the tenant database
                # System settings use empty string '' instead of NULL for UNIQUE constraint compatibility
                system_value = None
                with get_system_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        'SELECT value, is_encrypted FROM gui_settings WHERE key = ? AND owner_id = ?',
                        (key, '')
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        value, is_encrypted = row
                        if is_encrypted and value:
                            system_value = decrypt_value(value)
                        else:
                            system_value = self._parse_value(value)
                
                # Check tenant database for user-specific override
                tenant_value = None
                with get_tenant_db_connection(self.owner_id) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        'SELECT value, is_encrypted FROM gui_settings WHERE key = ? AND owner_id = ?',
                        (key, self.owner_id)
                    )
                    row = cursor.fetchone()
                    if row:
                        value, is_encrypted = row
                        if is_encrypted and value:
                            tenant_value = decrypt_value(value)
                        else:
                            tenant_value = self._parse_value(value)
                
                # For log_level specifically, apply hierarchical enforcement:
                # System log level acts as ceiling - tenant cannot set more verbose than system
                if key == "log_level" and system_value is not None and tenant_value is not None:
                    log_levels = {"none": 0, "error": 1, "debug": 2}
                    system_level = log_levels.get(system_value.lower() if isinstance(system_value, str) else "none", 0)
                    tenant_level = log_levels.get(tenant_value.lower() if isinstance(tenant_value, str) else "none", 0)
                    # Return the more restrictive (lower) level
                    return list(log_levels.keys())[list(log_levels.values()).index(min(system_level, tenant_level))]
                
                # For other keys, system value takes precedence if it exists
                if system_value is not None:
                    return system_value
                if tenant_value is not None:
                    return tenant_value
                return default
            else:
                with get_system_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Global setting (owner_id = '' for system settings)
                    cursor.execute(
                        'SELECT value, is_encrypted FROM gui_settings WHERE key = ? AND owner_id = ?',
                        (key, '')
                    )
                    row = cursor.fetchone()

                    if row:
                        value, is_encrypted = row
                        if is_encrypted and value:
                            decrypted = decrypt_value(value)
                            return decrypted if decrypted else default
                        return self._parse_value(value)
                    
                    return default
        except Exception:
            # If database is not available or table doesn't exist, return default
            return default

    def _parse_value(self, value: Any) -> Any:
        """Parse value from database."""
        if value is None:
            return None
        try:
            # Try to parse as JSON
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, return as string
            return value

    def get_ollama_base_url(self) -> str:
        """Get Ollama base URL from settings."""
        return self.get_setting("ollama_base_url", "http://localhost:11434")

    def get_parsing_model(self) -> str:
        """Get document parsing model from settings."""
        return self.get_setting("parsing_model", "grok")

    def get_chunking_model(self) -> str:
        """Get text chunking model from settings."""
        return self.get_setting("chunking_model", "grok")

    def get_classification_model(self) -> str:
        """Get document classification model from settings."""
        return self.get_setting("classification_model", "grok")

    def get_generation_model(self) -> str:
        """Get dataset generation model from settings."""
        return self.get_setting("generation_model", "grok")

    def get_taxonomy_provider(self) -> str:
        """Get taxonomy generation provider from settings."""
        return self.get_setting("taxonomy_provider", "grok")

    def get_parsing_ollama_model(self) -> str:
        """Get Ollama model for document parsing."""
        return self.get_setting("parsing_ollama_model", "benhaotang/Nanonets-OCR-s:latest")

    def get_parsing_gemini_model(self) -> str:
        """Get Gemini model for document parsing."""
        return self.get_setting("parsing_gemini_model", "gemini-2.5-flash-latest")

    def get_parsing_grok_model(self) -> str:
        """Get Grok model for document parsing."""
        return self.get_setting("parsing_grok_model", "grok-4-fast-non-reasoning")

    def get_parsing_openai_model(self) -> str:
        """Get OpenAI model for document parsing."""
        return self.get_setting("parsing_openai_model", "gpt-4.1-mini-2025-04-14")

    def get_chunking_ollama_model(self) -> str:
        """Get Ollama model for text chunking."""
        return self.get_setting("chunking_ollama_model", "deepseek-r1:7b")

    def get_chunking_gemini_model(self) -> str:
        """Get Gemini model for text chunking."""
        return self.get_setting("chunking_gemini_model", "gemini-2.5-flash-latest")

    def get_chunking_grok_model(self) -> str:
        """Get Grok model for text chunking."""
        return self.get_setting("chunking_grok_model", "grok-4-fast-reasoning")

    def get_chunking_openai_model(self) -> str:
        """Get OpenAI model for text chunking."""
        return self.get_setting("chunking_openai_model", "gpt-4.1-mini-2025-04-14")

    def get_chunking_ollama_num_ctx(self) -> int:
        """Get Ollama num_ctx parameter for text chunking."""
        return self.get_setting("chunking_ollama_num_ctx", 32768)

    # Ollama parameter getters for chunking
    def get_chunking_ollama_temperature(self) -> float:
        """Get Ollama temperature parameter for text chunking."""
        return self.get_setting("chunking_ollama_temperature", 0.1)

    def get_chunking_ollama_repeat_penalty(self) -> float:
        """Get Ollama repeat_penalty parameter for text chunking."""
        return self.get_setting("chunking_ollama_repeat_penalty", 1.1)

    def get_chunking_ollama_top_p(self) -> float:
        """Get Ollama top_p parameter for text chunking."""
        return self.get_setting("chunking_ollama_top_p", 0.9)

    def get_chunking_ollama_top_k(self) -> int:
        """Get Ollama top_k parameter for text chunking."""
        return self.get_setting("chunking_ollama_top_k", 20)

    def get_chunking_ollama_num_predict(self) -> int:
        """Get Ollama num_predict parameter for text chunking."""
        return self.get_setting("chunking_ollama_num_predict", 1024)

    def get_chunking_ollama_seed(self) -> Optional[int]:
        """Get Ollama seed parameter for text chunking."""
        return self.get_setting("chunking_ollama_seed", None)

    def get_classification_ollama_model(self) -> str:
        """Get Ollama model for document classification."""
        return self.get_setting("classification_ollama_model", "deepseek-r1:7b")

    def get_classification_gemini_model(self) -> str:
        """Get Gemini model for document classification."""
        return self.get_setting("classification_gemini_model", "gemini-2.5-flash-latest")

    def get_classification_grok_model(self) -> str:
        """Get Grok model for document classification."""
        return self.get_setting("classification_grok_model", "grok-4-fast-reasoning")

    def get_classification_openai_model(self) -> str:
        """Get OpenAI model for document classification."""
        return self.get_setting("classification_openai_model", "gpt-4.1-mini-2025-04-14")

    # Ollama parameter getters for classification
    def get_classification_ollama_temperature(self) -> float:
        """Get Ollama temperature parameter for document classification."""
        return self.get_setting("classification_ollama_temperature", 0.1)

    def get_classification_ollama_repeat_penalty(self) -> float:
        """Get Ollama repeat_penalty parameter for document classification."""
        return self.get_setting("classification_ollama_repeat_penalty", 1.1)

    def get_classification_ollama_top_p(self) -> float:
        """Get Ollama top_p parameter for document classification."""
        return self.get_setting("classification_ollama_top_p", 0.9)

    def get_classification_ollama_top_k(self) -> int:
        """Get Ollama top_k parameter for document classification."""
        return self.get_setting("classification_ollama_top_k", 20)

    def get_classification_ollama_num_predict(self) -> int:
        """Get Ollama num_predict parameter for document classification."""
        return self.get_setting("classification_ollama_num_predict", 1024)

    def get_classification_ollama_seed(self) -> Optional[int]:
        """Get Ollama seed parameter for document classification."""
        return self.get_setting("classification_ollama_seed", None)

    def get_classification_ollama_num_ctx(self) -> int:
        """Get Ollama num_ctx parameter for document classification."""
        return self.get_setting("classification_ollama_num_ctx", 32768)

    def get_generation_ollama_model(self) -> str:
        """Get Ollama model for dataset generation."""
        return self.get_setting("generation_ollama_model", "deepseek-r1:7b")

    def get_generation_gemini_model(self) -> str:
        """Get Gemini model for dataset generation."""
        return self.get_setting("generation_gemini_model", "gemini-2.5-flash-latest")

    def get_generation_grok_model(self) -> str:
        """Get Grok model for dataset generation."""
        return self.get_setting("generation_grok_model", "grok-4-fast-reasoning")

    def get_generation_openai_model(self) -> str:
        """Get OpenAI model for dataset generation."""
        return self.get_setting("generation_openai_model", "gpt-4.1-mini-2025-04-14")

    # Ollama parameter getters for generation
    def get_generation_ollama_temperature(self) -> float:
        """Get Ollama temperature parameter for dataset generation."""
        return self.get_setting("generation_ollama_temperature", 0.1)

    def get_generation_ollama_repeat_penalty(self) -> float:
        """Get Ollama repeat_penalty parameter for dataset generation."""
        return self.get_setting("generation_ollama_repeat_penalty", 1.1)

    def get_generation_ollama_top_p(self) -> float:
        """Get Ollama top_p parameter for dataset generation."""
        return self.get_setting("generation_ollama_top_p", 0.9)

    def get_generation_ollama_top_k(self) -> int:
        """Get Ollama top_k parameter for dataset generation."""
        return self.get_setting("generation_ollama_top_k", 20)

    def get_generation_ollama_num_predict(self) -> int:
        """Get Ollama num_predict parameter for dataset generation."""
        return self.get_setting("generation_ollama_num_predict", 4096)

    def get_generation_ollama_seed(self) -> Optional[int]:
        """Get Ollama seed parameter for dataset generation."""
        return self.get_setting("generation_ollama_seed", None)

    def get_generation_ollama_num_ctx(self) -> int:
        """Get Ollama num_ctx parameter for dataset generation."""
        return self.get_setting("generation_ollama_num_ctx", 32768)

    def get_quality_ollama_model(self) -> str:
        """Get Ollama model for dataset quality analysis."""
        return self.get_setting("quality_ollama_model", "deepseek-r1:7b")

    def get_quality_gemini_model(self) -> str:
        """Get Gemini model for dataset quality analysis."""
        return self.get_setting("quality_gemini_model", "gemini-2.5-flash-latest")

    def get_quality_grok_model(self) -> str:
        """Get Grok model for dataset quality analysis."""
        return self.get_setting("quality_grok_model", "grok-4-fast-reasoning")

    def get_quality_openai_model(self) -> str:
        """Get OpenAI model for dataset quality analysis."""
        return self.get_setting("quality_openai_model", "gpt-4.1-mini-2025-04-14")

    # Ollama parameter getters for quality
    def get_quality_ollama_temperature(self) -> float:
        """Get Ollama temperature parameter for dataset quality analysis."""
        return self.get_setting("quality_ollama_temperature", 0.1)

    def get_quality_ollama_repeat_penalty(self) -> float:
        """Get Ollama repeat_penalty parameter for dataset quality analysis."""
        return self.get_setting("quality_ollama_repeat_penalty", 1.1)

    def get_quality_ollama_top_p(self) -> float:
        """Get Ollama top_p parameter for dataset quality analysis."""
        return self.get_setting("quality_ollama_top_p", 0.9)

    def get_quality_ollama_top_k(self) -> int:
        """Get Ollama top_k parameter for dataset quality analysis."""
        return self.get_setting("quality_ollama_top_k", 20)

    def get_quality_ollama_num_predict(self) -> int:
        """Get Ollama num_predict parameter for dataset quality analysis."""
        return self.get_setting("quality_ollama_num_predict", 1024)

    def get_quality_ollama_seed(self) -> Optional[int]:
        """Get Ollama seed parameter for dataset quality analysis."""
        return self.get_setting("quality_ollama_seed", None)

    def get_quality_ollama_num_ctx(self) -> int:
        """Get Ollama num_ctx parameter for dataset quality analysis."""
        return self.get_setting("quality_ollama_num_ctx", 32768)

    def get_taxonomy_ollama_model(self) -> str:
        """Get Ollama model for taxonomy generation."""
        return self.get_setting("taxonomy_ollama_model", "deepseek-r1:7b")

    def get_taxonomy_gemini_model(self) -> str:
        """Get Gemini model for taxonomy generation."""
        return self.get_setting("taxonomy_gemini_model", "gemini-2.5-flash-latest")

    def get_taxonomy_grok_model(self) -> str:
        """Get Grok model for taxonomy generation."""
        return self.get_setting("taxonomy_grok_model", "grok-4-fast-reasoning")

    def get_taxonomy_openai_model(self) -> str:
        """Get OpenAI model for taxonomy generation."""
        return self.get_setting("taxonomy_openai_model", "gpt-4.1-mini-2025-04-14")

    # Ollama parameter getters for parsing
    def get_parsing_ollama_temperature(self) -> float:
        """Get Ollama temperature parameter for document parsing."""
        return self.get_setting("parsing_ollama_temperature", 0.1)

    def get_parsing_ollama_repeat_penalty(self) -> float:
        """Get Ollama repeat_penalty parameter for document parsing."""
        return self.get_setting("parsing_ollama_repeat_penalty", 1.2)

    def get_parsing_ollama_top_p(self) -> float:
        """Get Ollama top_p parameter for document parsing."""
        return self.get_setting("parsing_ollama_top_p", 0.9)

    def get_parsing_ollama_top_k(self) -> int:
        """Get Ollama top_k parameter for document parsing."""
        return self.get_setting("parsing_ollama_top_k", 20)

    def get_parsing_ollama_num_predict(self) -> int:
        """Get Ollama num_predict parameter for document parsing."""
        return self.get_setting("parsing_ollama_num_predict", 1024)

    def get_parsing_ollama_seed(self) -> Optional[int]:
        """Get Ollama seed parameter for document parsing."""
        return self.get_setting("parsing_ollama_seed", None)

    def get_parsing_ollama_num_ctx(self) -> int:
        """Get Ollama num_ctx parameter for document parsing."""
        return self.get_setting("parsing_ollama_num_ctx", 32768)

    # Ollama parameter getters for taxonomy
    def get_taxonomy_ollama_temperature(self) -> float:
        """Get Ollama temperature parameter for taxonomy generation."""
        return self.get_setting("taxonomy_ollama_temperature", 0.1)

    def get_taxonomy_ollama_repeat_penalty(self) -> float:
        """Get Ollama repeat_penalty parameter for taxonomy generation."""
        return self.get_setting("taxonomy_ollama_repeat_penalty", 1.1)

    def get_taxonomy_ollama_top_p(self) -> float:
        """Get Ollama top_p parameter for taxonomy generation."""
        return self.get_setting("taxonomy_ollama_top_p", 0.9)

    def get_taxonomy_ollama_top_k(self) -> int:
        """Get Ollama top_k parameter for taxonomy generation."""
        return self.get_setting("taxonomy_ollama_top_k", 20)

    def get_taxonomy_ollama_num_predict(self) -> int:
        """Get Ollama num_predict parameter for taxonomy generation."""
        return self.get_setting("taxonomy_ollama_num_predict", 1024)

    def get_taxonomy_ollama_seed(self) -> Optional[int]:
        """Get Ollama seed parameter for taxonomy generation."""
        return self.get_setting("taxonomy_ollama_seed", None)

    def get_taxonomy_ollama_num_ctx(self) -> int:
        """Get Ollama num_ctx parameter for taxonomy generation."""
        return self.get_setting("taxonomy_ollama_num_ctx", 32768)

    def get_global_max_concurrent_jobs(self) -> int:
        """Get global max concurrent jobs from settings."""
        return self.get_setting("max_concurrent_jobs", 6)

    def get_max_concurrent_jobs_per_user(self) -> int:
        """Get max concurrent jobs per user from settings."""
        return self.get_setting("max_concurrent_jobs_per_user", 6)

    def get_system_max_concurrent_jobs(self) -> int:
        """Get system-wide hard ceiling for max concurrent jobs."""
        return self.get_setting("system_max_concurrent_jobs", 10)

    def get_system_max_concurrent_jobs_per_user(self) -> int:
        """Get system-wide hard ceiling for per-user concurrent jobs."""
        return self.get_setting("system_max_concurrent_jobs_per_user", 10)

    def get_quality_threshold(self) -> float:
        """Get quality threshold from settings."""
        return self.get_setting("quality_threshold", 0.7)

    def get_max_file_size_mb(self) -> int:
        """Get max file size in MB from settings."""
        return self.get_setting("max_file_size_mb", 200)

    def get_theme(self) -> str:
        """Get theme from settings."""
        return self.get_setting("theme", "Light")

    def get_per_user_max_concurrent_jobs(self) -> int:
        """Get per-user max concurrent jobs from settings."""
        return self.get_setting("max_concurrent_jobs_per_user", 3)

    # Benchmarking model getters
    def get_benchmarking_ollama_model(self) -> str:
        """Get Ollama model for benchmarking."""
        return self.get_setting("benchmarking_ollama_model", "deepseek-r1:7b")

    def get_benchmarking_gemini_model(self) -> str:
        """Get Gemini model for benchmarking."""
        return self.get_setting("benchmarking_gemini_model", "gemini-2.5-flash-latest")

    def get_benchmarking_grok_model(self) -> str:
        """Get Grok model for benchmarking."""
        return self.get_setting("benchmarking_grok_model", "grok-4-fast-reasoning")

    def get_benchmarking_openai_model(self) -> str:
        """Get OpenAI model for benchmarking."""
        return self.get_setting("benchmarking_openai_model", "gpt-4.1-mini-2025-04-14")

    # Ollama parameter getters for benchmarking
    def get_benchmarking_ollama_temperature(self) -> float:
        """Get Ollama temperature parameter for benchmarking."""
        return self.get_setting("benchmarking_ollama_temperature", 0.1)

    def get_benchmarking_ollama_repeat_penalty(self) -> float:
        """Get Ollama repeat_penalty parameter for benchmarking."""
        return self.get_setting("benchmarking_ollama_repeat_penalty", 1.2)

    def get_benchmarking_ollama_top_p(self) -> float:
        """Get Ollama top_p parameter for benchmarking."""
        return self.get_setting("benchmarking_ollama_top_p", 0.9)

    def get_benchmarking_ollama_top_k(self) -> int:
        """Get Ollama top_k parameter for benchmarking."""
        return self.get_setting("benchmarking_ollama_top_k", 20)

    def get_benchmarking_ollama_num_predict(self) -> int:
        """Get Ollama num_predict parameter for benchmarking."""
        return self.get_setting("benchmarking_ollama_num_predict", 1024)

    def get_benchmarking_ollama_seed(self) -> Optional[int]:
        """Get Ollama seed parameter for benchmarking."""
        return self.get_setting("benchmarking_ollama_seed", None)

    def get_benchmarking_ollama_num_ctx(self) -> int:
        """Get Ollama num_ctx parameter for benchmarking."""
        return self.get_setting("benchmarking_ollama_num_ctx", 32768)

    def get_log_level(self) -> LogLevel:
        """Get the current log level from settings."""
        level = self.get_setting("log_level", LogLevel.NONE.value)
        try:
            return LogLevel(level.lower())
        except (ValueError, AttributeError):
            return LogLevel.NONE

    def get_job_history_retention_days(self) -> int:
        """
        Get the job history retention period in days.
        
        Returns:
            Number of days to retain job history entries (default: 90)
        """
        return self.get_setting("job_history_retention_days", 90)

    def set_job_history_retention_days(self, days: int) -> bool:
        """
        Set the job history retention period in days.
        
        Args:
            days: Number of days to retain job history (must be 1-365)
            
        Returns:
            True if successful, False otherwise
        """
        if not isinstance(days, int) or days < 1 or days > 365:
            return False
        return self.set_setting("job_history_retention_days", days)

    def set_log_level(self, level: LogLevel) -> bool:
        """Set the global log level in settings."""
        return self.set_setting("log_level", level.value)

    def get_license_status(self) -> str:
        """Get the license status from settings."""
        return self.get_setting("license_status", "inactive")

    def set_setting(self, key: str, value: Any) -> bool:
        """
        Set a setting value in the GUI settings database.
        Automatically encrypts sensitive values.
        
        Args:
            key: Setting key
            value: Value to set (will be JSON serialized)
            
        Returns:
            True if successful, False otherwise
        """
        # Import here to avoid circular imports
        from ..storage.src.database import get_tenant_db_connection, get_system_db_connection
        
        # Determine if this value should be encrypted
        is_encrypted = is_sensitive_key(key)
        
        # Encrypt if needed
        if is_encrypted and value:
            if isinstance(value, str):
                value = encrypt_value(value)
            else:
                value = encrypt_value(str(value))
        
        # JSON serialize non-encrypted values
        if not is_encrypted:
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
        else:
            serialized_value = value
        
        try:
            # Use empty string '' for system settings (owner_id is None means system-wide setting)
            # This ensures UNIQUE constraint works properly in SQLite (NULL != NULL issue)
            if self.owner_id:
                with get_tenant_db_connection(self.owner_id) as conn:
                    cursor = conn.cursor()
                    
                    # Insert or replace setting
                    cursor.execute('''
                        INSERT OR REPLACE INTO gui_settings (key, value, is_encrypted, owner_id, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (key, serialized_value, is_encrypted, self.owner_id))
                    
                    conn.commit()
                    return True
            else:
                with get_system_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Global setting (owner_id = '' for system settings)
                    cursor.execute('''
                        INSERT OR REPLACE INTO gui_settings (key, value, is_encrypted, owner_id, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (key, serialized_value, is_encrypted, ''))
                    
                    conn.commit()
                    return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Error setting {key}: {e}")
            return False


# Global instance (backwards compatibility)
backend_settings = BackendSettings()
