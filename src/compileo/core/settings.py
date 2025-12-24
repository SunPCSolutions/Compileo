"""
Backend settings accessor for Compileo.
Reads settings from the GUI settings database.
"""

from typing import Any, Optional
import json
from enum import Enum
from ..storage.src.database import get_db_connection


class LogLevel(str, Enum):
    """Log level options."""
    NONE = "none"
    ERROR = "error"
    DEBUG = "debug"


class BackendSettings:
    """Access GUI settings from the backend."""

    @staticmethod
    def get_openai_api_key() -> str:
        """Get OpenAI API key from settings."""
        return BackendSettings.get_setting("openai_api_key", "")

    @staticmethod
    def get_huggingface_hub_token() -> str:
        """Get Hugging Face Hub token from settings."""
        # Use the key that the GUI actually saves: 'huggingface_api_key'
        return BackendSettings.get_setting("huggingface_api_key", "")

    @staticmethod
    def get_setting(key: str, default: Any = None) -> Any:
        """Get a setting value from the GUI settings database.

        Args:
            key: Setting key
            default: Default value if key not found or database unavailable

        Returns:
            Setting value or default
        """
        conn = None
        try:
            conn = get_db_connection()
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
        except Exception:
            # If database is not available or table doesn't exist, return default
            return default
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    @staticmethod
    def get_ollama_base_url() -> str:
        """Get Ollama base URL from settings."""
        return BackendSettings.get_setting("ollama_base_url", "http://localhost:11434")

    @staticmethod
    def get_parsing_model() -> str:
        """Get document parsing model from settings."""
        return BackendSettings.get_setting("parsing_model", "grok")

    @staticmethod
    def get_chunking_model() -> str:
        """Get text chunking model from settings."""
        return BackendSettings.get_setting("chunking_model", "grok")

    @staticmethod
    def get_classification_model() -> str:
        """Get document classification model from settings."""
        return BackendSettings.get_setting("classification_model", "grok")

    @staticmethod
    def get_generation_model() -> str:
        """Get dataset generation model from settings."""
        return BackendSettings.get_setting("generation_model", "grok")

    @staticmethod
    def get_taxonomy_provider() -> str:
        """Get taxonomy generation provider from settings."""
        return BackendSettings.get_setting("taxonomy_provider", "grok")

    @staticmethod
    def get_parsing_ollama_model() -> str:
        """Get Ollama model for document parsing."""
        return BackendSettings.get_setting("parsing_ollama_model", "benhaotang/Nanonets-OCR-s:latest")

    @staticmethod
    def get_parsing_gemini_model() -> str:
        """Get Gemini model for document parsing."""
        return BackendSettings.get_setting("parsing_gemini_model", "gemini-2.5-flash-latest")

    @staticmethod
    def get_parsing_grok_model() -> str:
        """Get Grok model for document parsing."""
        return BackendSettings.get_setting("parsing_grok_model", "grok-4-fast-non-reasoning")

    @staticmethod
    def get_parsing_openai_model() -> str:
        """Get OpenAI model for document parsing."""
        return BackendSettings.get_setting("parsing_openai_model", "gpt-4.1-mini-2025-04-14")

    @staticmethod
    def get_chunking_ollama_model() -> str:
        """Get Ollama model for text chunking."""
        return BackendSettings.get_setting("chunking_ollama_model", "deepseek-r1:7b")

    @staticmethod
    def get_chunking_gemini_model() -> str:
        """Get Gemini model for text chunking."""
        return BackendSettings.get_setting("chunking_gemini_model", "gemini-2.5-flash-latest")

    @staticmethod
    def get_chunking_grok_model() -> str:
        """Get Grok model for text chunking."""
        return BackendSettings.get_setting("chunking_grok_model", "grok-4-fast-reasoning")

    @staticmethod
    def get_chunking_openai_model() -> str:
        """Get OpenAI model for text chunking."""
        return BackendSettings.get_setting("chunking_openai_model", "gpt-4.1-mini-2025-04-14")

    @staticmethod
    def get_chunking_ollama_num_ctx() -> int:
        """Get Ollama num_ctx parameter for text chunking."""
        return BackendSettings.get_setting("chunking_ollama_num_ctx", 32768)

    # Ollama parameter getters for chunking
    @staticmethod
    def get_chunking_ollama_temperature() -> float:
        """Get Ollama temperature parameter for text chunking."""
        return BackendSettings.get_setting("chunking_ollama_temperature", 0.1)

    @staticmethod
    def get_chunking_ollama_repeat_penalty() -> float:
        """Get Ollama repeat_penalty parameter for text chunking."""
        return BackendSettings.get_setting("chunking_ollama_repeat_penalty", 1.1)

    @staticmethod
    def get_chunking_ollama_top_p() -> float:
        """Get Ollama top_p parameter for text chunking."""
        return BackendSettings.get_setting("chunking_ollama_top_p", 0.9)

    @staticmethod
    def get_chunking_ollama_top_k() -> int:
        """Get Ollama top_k parameter for text chunking."""
        return BackendSettings.get_setting("chunking_ollama_top_k", 20)

    @staticmethod
    def get_chunking_ollama_num_predict() -> int:
        """Get Ollama num_predict parameter for text chunking."""
        return BackendSettings.get_setting("chunking_ollama_num_predict", 1024)

    @staticmethod
    def get_chunking_ollama_seed() -> Optional[int]:
        """Get Ollama seed parameter for text chunking."""
        return BackendSettings.get_setting("chunking_ollama_seed", None)

    @staticmethod
    def get_classification_ollama_model() -> str:
        """Get Ollama model for document classification."""
        return BackendSettings.get_setting("classification_ollama_model", "deepseek-r1:7b")

    @staticmethod
    def get_classification_gemini_model() -> str:
        """Get Gemini model for document classification."""
        return BackendSettings.get_setting("classification_gemini_model", "gemini-2.5-flash-latest")

    @staticmethod
    def get_classification_grok_model() -> str:
        """Get Grok model for document classification."""
        return BackendSettings.get_setting("classification_grok_model", "grok-4-fast-reasoning")

    @staticmethod
    def get_classification_openai_model() -> str:
        """Get OpenAI model for document classification."""
        return BackendSettings.get_setting("classification_openai_model", "gpt-4.1-mini-2025-04-14")

    @staticmethod
    def get_generation_ollama_model() -> str:
        """Get Ollama model for dataset generation."""
        return BackendSettings.get_setting("generation_ollama_model", "deepseek-r1:7b")

    @staticmethod
    def get_generation_gemini_model() -> str:
        """Get Gemini model for dataset generation."""
        return BackendSettings.get_setting("generation_gemini_model", "gemini-2.5-flash-latest")

    @staticmethod
    def get_generation_grok_model() -> str:
        """Get Grok model for dataset generation."""
        return BackendSettings.get_setting("generation_grok_model", "grok-4-fast-reasoning")

    @staticmethod
    def get_generation_openai_model() -> str:
        """Get OpenAI model for dataset generation."""
        return BackendSettings.get_setting("generation_openai_model", "gpt-4.1-mini-2025-04-14")

    @staticmethod
    def get_quality_ollama_model() -> str:
        """Get Ollama model for dataset quality analysis."""
        return BackendSettings.get_setting("quality_ollama_model", "deepseek-r1:7b")

    @staticmethod
    def get_quality_gemini_model() -> str:
        """Get Gemini model for dataset quality analysis."""
        return BackendSettings.get_setting("quality_gemini_model", "gemini-2.5-flash-latest")

    @staticmethod
    def get_quality_grok_model() -> str:
        """Get Grok model for dataset quality analysis."""
        return BackendSettings.get_setting("quality_grok_model", "grok-4-fast-reasoning")

    @staticmethod
    def get_quality_openai_model() -> str:
        """Get OpenAI model for dataset quality analysis."""
        return BackendSettings.get_setting("quality_openai_model", "gpt-4.1-mini-2025-04-14")

    @staticmethod
    def get_taxonomy_ollama_model() -> str:
        """Get Ollama model for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_ollama_model", "deepseek-r1:7b")

    @staticmethod
    def get_taxonomy_gemini_model() -> str:
        """Get Gemini model for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_gemini_model", "gemini-2.5-flash-latest")

    @staticmethod
    def get_taxonomy_grok_model() -> str:
        """Get Grok model for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_grok_model", "grok-4-fast-reasoning")

    @staticmethod
    def get_taxonomy_openai_model() -> str:
        """Get OpenAI model for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_openai_model", "gpt-4.1-mini-2025-04-14")

    @staticmethod
    def get_taxonomy_depth() -> int:
        """Get taxonomy hierarchy depth from settings."""
        return BackendSettings.get_setting("taxonomy_depth", 3)

    @staticmethod
    def get_sample_size() -> int:
        """Get taxonomy sample size from settings."""
        return BackendSettings.get_setting("sample_size", 100)

    # Ollama parameter getters for parsing
    @staticmethod
    def get_parsing_ollama_temperature() -> float:
        """Get Ollama temperature parameter for document parsing."""
        return BackendSettings.get_setting("parsing_ollama_temperature", 0.1)

    @staticmethod
    def get_parsing_ollama_repeat_penalty() -> float:
        """Get Ollama repeat_penalty parameter for document parsing."""
        return BackendSettings.get_setting("parsing_ollama_repeat_penalty", 1.2)

    @staticmethod
    def get_parsing_ollama_top_p() -> float:
        """Get Ollama top_p parameter for document parsing."""
        return BackendSettings.get_setting("parsing_ollama_top_p", 0.9)

    @staticmethod
    def get_parsing_ollama_top_k() -> int:
        """Get Ollama top_k parameter for document parsing."""
        return BackendSettings.get_setting("parsing_ollama_top_k", 20)

    @staticmethod
    def get_parsing_ollama_num_predict() -> int:
        """Get Ollama num_predict parameter for document parsing."""
        return BackendSettings.get_setting("parsing_ollama_num_predict", 1024)

    @staticmethod
    def get_parsing_ollama_seed() -> Optional[int]:
        """Get Ollama seed parameter for document parsing."""
        return BackendSettings.get_setting("parsing_ollama_seed", None)

    @staticmethod
    def get_parsing_ollama_num_ctx() -> int:
        """Get Ollama num_ctx parameter for document parsing."""
        return BackendSettings.get_setting("parsing_ollama_num_ctx", 32768)

    # Ollama parameter getters for taxonomy
    @staticmethod
    def get_taxonomy_ollama_temperature() -> float:
        """Get Ollama temperature parameter for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_ollama_temperature", 0.1)

    @staticmethod
    def get_taxonomy_ollama_repeat_penalty() -> float:
        """Get Ollama repeat_penalty parameter for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_ollama_repeat_penalty", 1.1)

    @staticmethod
    def get_taxonomy_ollama_top_p() -> float:
        """Get Ollama top_p parameter for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_ollama_top_p", 0.9)

    @staticmethod
    def get_taxonomy_ollama_top_k() -> int:
        """Get Ollama top_k parameter for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_ollama_top_k", 40)

    @staticmethod
    def get_taxonomy_ollama_num_predict() -> int:
        """Get Ollama num_predict parameter for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_ollama_num_predict", 1024)

    @staticmethod
    def get_taxonomy_ollama_seed() -> Optional[int]:
        """Get Ollama seed parameter for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_ollama_seed", None)

    @staticmethod
    def get_taxonomy_ollama_num_ctx() -> int:
        """Get Ollama num_ctx parameter for taxonomy generation."""
        return BackendSettings.get_setting("taxonomy_ollama_num_ctx", 32768)

    # Ollama parameter getters for classification
    @staticmethod
    def get_classification_ollama_temperature() -> float:
        """Get Ollama temperature parameter for document classification."""
        return BackendSettings.get_setting("classification_ollama_temperature", 0.1)

    @staticmethod
    def get_classification_ollama_repeat_penalty() -> float:
        """Get Ollama repeat_penalty parameter for document classification."""
        return BackendSettings.get_setting("classification_ollama_repeat_penalty", 1.1)

    @staticmethod
    def get_classification_ollama_top_p() -> float:
        """Get Ollama top_p parameter for document classification."""
        return BackendSettings.get_setting("classification_ollama_top_p", 0.9)

    @staticmethod
    def get_classification_ollama_top_k() -> int:
        """Get Ollama top_k parameter for document classification."""
        return BackendSettings.get_setting("classification_ollama_top_k", 40)

    @staticmethod
    def get_classification_ollama_num_predict() -> int:
        """Get Ollama num_predict parameter for document classification."""
        return BackendSettings.get_setting("classification_ollama_num_predict", 1024)

    @staticmethod
    def get_classification_ollama_seed() -> Optional[int]:
        """Get Ollama seed parameter for document classification."""
        return BackendSettings.get_setting("classification_ollama_seed", None)

    @staticmethod
    def get_classification_ollama_num_ctx() -> int:
        """Get Ollama num_ctx parameter for document classification."""
        return BackendSettings.get_setting("classification_ollama_num_ctx", 32768)

    # Ollama parameter getters for generation
    @staticmethod
    def get_generation_ollama_temperature() -> float:
        """Get Ollama temperature parameter for dataset generation."""
        return BackendSettings.get_setting("generation_ollama_temperature", 0.1)

    @staticmethod
    def get_generation_ollama_repeat_penalty() -> float:
        """Get Ollama repeat_penalty parameter for dataset generation."""
        return BackendSettings.get_setting("generation_ollama_repeat_penalty", 1.1)

    @staticmethod
    def get_generation_ollama_top_p() -> float:
        """Get Ollama top_p parameter for dataset generation."""
        return BackendSettings.get_setting("generation_ollama_top_p", 0.9)

    @staticmethod
    def get_generation_ollama_top_k() -> int:
        """Get Ollama top_k parameter for dataset generation."""
        return BackendSettings.get_setting("generation_ollama_top_k", 40)

    @staticmethod
    def get_generation_ollama_num_predict() -> int:
        """Get Ollama num_predict parameter for dataset generation."""
        return BackendSettings.get_setting("generation_ollama_num_predict", 1024)

    @staticmethod
    def get_generation_ollama_seed() -> Optional[int]:
        """Get Ollama seed parameter for dataset generation."""
        return BackendSettings.get_setting("generation_ollama_seed", None)

    @staticmethod
    def get_generation_ollama_num_ctx() -> int:
        """Get Ollama num_ctx parameter for dataset generation."""
        return BackendSettings.get_setting("generation_ollama_num_ctx", 32768)

    @staticmethod
    def get_global_max_concurrent_jobs() -> int:
        """Get global max concurrent jobs from settings."""
        return BackendSettings.get_setting("max_concurrent_jobs", 5)

    @staticmethod
    def get_max_concurrent_jobs_per_user() -> int:
        """Get max concurrent jobs per user from settings."""
        return BackendSettings.get_setting("max_concurrent_jobs_per_user", 5)

    @staticmethod
    def get_quality_threshold() -> float:
        """Get quality threshold from settings."""
        return BackendSettings.get_setting("quality_threshold", 0.7)

    @staticmethod
    def get_max_file_size_mb() -> int:
        """Get max file size in MB from settings."""
        return BackendSettings.get_setting("max_file_size_mb", 200)

    @staticmethod
    def get_theme() -> str:
        """Get theme from settings."""
        return BackendSettings.get_setting("theme", "Light")

    @staticmethod
    def get_default_page_size() -> int:
        """Get default page size from settings."""
        return BackendSettings.get_setting("default_page_size", 50)

    # TODO: Uncomment when multi-user architecture is implemented
    # @staticmethod
    # def get_per_user_max_concurrent_jobs() -> int:
    #     """Get per-user max concurrent jobs from settings."""
    #     return BackendSettings.get_setting("max_concurrent_jobs_per_user", 3)

    # Benchmarking model getters
    @staticmethod
    def get_benchmarking_ollama_model() -> str:
        """Get Ollama model for benchmarking."""
        return BackendSettings.get_setting("benchmarking_ollama_model", "deepseek-r1:7b")

    @staticmethod
    def get_benchmarking_gemini_model() -> str:
        """Get Gemini model for benchmarking."""
        return BackendSettings.get_setting("benchmarking_gemini_model", "gemini-2.5-flash-latest")

    @staticmethod
    def get_benchmarking_grok_model() -> str:
        """Get Grok model for benchmarking."""
        return BackendSettings.get_setting("benchmarking_grok_model", "grok-4-fast-reasoning")

    @staticmethod
    def get_benchmarking_openai_model() -> str:
        """Get OpenAI model for benchmarking."""
        return BackendSettings.get_setting("benchmarking_openai_model", "gpt-4.1-mini-2025-04-14")

    # Ollama parameter getters for benchmarking
    @staticmethod
    def get_benchmarking_ollama_temperature() -> float:
        """Get Ollama temperature parameter for benchmarking."""
        return BackendSettings.get_setting("benchmarking_ollama_temperature", 0.1)

    @staticmethod
    def get_benchmarking_ollama_repeat_penalty() -> float:
        """Get Ollama repeat_penalty parameter for benchmarking."""
        return BackendSettings.get_setting("benchmarking_ollama_repeat_penalty", 1.2)

    @staticmethod
    def get_benchmarking_ollama_top_p() -> float:
        """Get Ollama top_p parameter for benchmarking."""
        return BackendSettings.get_setting("benchmarking_ollama_top_p", 0.9)

    @staticmethod
    def get_benchmarking_ollama_top_k() -> int:
        """Get Ollama top_k parameter for benchmarking."""
        return BackendSettings.get_setting("benchmarking_ollama_top_k", 20)

    @staticmethod
    def get_benchmarking_ollama_num_predict() -> int:
        """Get Ollama num_predict parameter for benchmarking."""
        return BackendSettings.get_setting("benchmarking_ollama_num_predict", 1024)

    @staticmethod
    def get_benchmarking_ollama_seed() -> Optional[int]:
        """Get Ollama seed parameter for benchmarking."""
        return BackendSettings.get_setting("benchmarking_ollama_seed", None)

    @staticmethod
    def get_benchmarking_ollama_num_ctx() -> int:
        """Get Ollama num_ctx parameter for benchmarking."""
        return BackendSettings.get_setting("benchmarking_ollama_num_ctx", 32768)

    @staticmethod
    def get_log_level() -> LogLevel:
        """Get the current log level from settings."""
        level = BackendSettings.get_setting("log_level", LogLevel.NONE.value)
        try:
            return LogLevel(level.lower())
        except (ValueError, AttributeError):
            return LogLevel.NONE

    @staticmethod
    def set_log_level(level: LogLevel) -> bool:
        """Set the log level in settings."""
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Use same serialization logic as SettingsStorage
            serialized_value = level.value if isinstance(level, LogLevel) else str(level)
            
            cursor.execute('''
                INSERT OR REPLACE INTO gui_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', ("log_level", json.dumps(serialized_value)))
            
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass


# Global instance
backend_settings = BackendSettings()