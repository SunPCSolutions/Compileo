"""Configuration settings for the Compileo API."""

import os
from typing import List, Optional

from pydantic import BaseModel
from ...core.settings import BackendSettings


class Settings(BaseModel):
    """Application settings loaded from environment variables and GUI settings."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    # Trusted Hosts
    allowed_hosts: List[str] = ["*"]

    # Authentication
    api_key_header: str = "X-API-Key"
    api_keys: List[str] = []  # Consolidated list of valid keys from Env, CLI, and DB
    cli_api_key_override: Optional[str] = None  # Key provided via command line

    # Database
    database_url: str = "sqlite:///database.db"

    # Redis (required for job queuing) - get from GUI settings or environment
    redis_url: str = "redis://localhost:6379/0"

    # File Storage
    upload_directory: str = "uploads"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB

    # Rate Limiting
    rate_limit_requests: int = 200
    rate_limit_window: int = 60  # seconds

    # Job Handling Limits
    global_max_concurrent_jobs: int = 10  # Global limit for all concurrent jobs
    # per_user_max_concurrent_jobs: int = 3  # Per-user concurrent job limit - TODO: Uncomment when multi-user architecture is implemented

    # Logging
    log_level: str = "INFO"

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Always try to load from environment, overriding default
        redis_host = os.getenv("COMPILEO_REDIS_HOST")
        redis_port = os.getenv("COMPILEO_REDIS_PORT")
        env_redis_url = os.getenv("REDIS_URL")
        
        if redis_host and redis_port:
            # Docker environment with specific host/port
            self.redis_url = f"redis://{redis_host}:{redis_port}/0"
        elif env_redis_url:
            # Explicitly provided URL (e.g. from .env file)
            self.redis_url = env_redis_url
        else:
            # Fallback to localhost for local development without Docker/Env
            self.redis_url = "redis://localhost:6379/0"
        
        # Ensure proper str type
        if self.redis_url is None:
            self.redis_url = "redis://localhost:6379/0"
        
        # Ensure redis_url uses 'redis' protocol
        if not self.redis_url.startswith("redis://"):
             self.redis_url = f"redis://{self.redis_url}"

        # Initial load
        self.api_keys = []
        # We will reload in the middleware

        # Override CORS origins from environment
        cors_origins_env = os.getenv("CORS_ORIGINS", "")
        if cors_origins_env:
            self.cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

        # Override job limits from GUI settings
        backend_settings = BackendSettings()
        self.global_max_concurrent_jobs = backend_settings.get_global_max_concurrent_jobs()
        # TODO: Uncomment when multi-user architecture is implemented
        # self.per_user_max_concurrent_jobs = backend_settings.get_per_user_max_concurrent_jobs()
        # self.per_user_max_concurrent_jobs = self.global_max_concurrent_jobs  # Single-user mode
        # TODO: Uncomment when multi-user architecture is implemented
        # self.per_user_max_concurrent_jobs = self.global_max_concurrent_jobs  # Single-user mode - TODO: Remove when multi-user implemented


    def set_cli_api_key(self, key: str):
        """Set the API key provided via CLI."""
        self.cli_api_key_override = key
        self.reload_api_keys()

    def reload_api_keys(self):
        """
        Reload and consolidate API keys from all sources:
        1. Environment Variables (COMPILEO_API_KEY, COMPILEO_API_KEYS)
        2. CLI Argument (--api-key)
        3. Database (gui_settings table)
        """
        all_keys = []

        # 1. Environment Variables
        import os
        api_key_single = os.getenv("COMPILEO_API_KEY")
        api_keys_multi = os.getenv("COMPILEO_API_KEYS")

        if api_key_single:
            val = api_key_single.strip()
            if val and val not in all_keys:
                all_keys.append(val)
        
        if api_keys_multi:
            multi_keys = [k.strip() for k in api_keys_multi.split(",") if k.strip()]
            for mk in multi_keys:
                if mk and mk not in all_keys:
                    all_keys.append(mk)

        # 2. CLI Argument
        if self.cli_api_key_override:
            val = self.cli_api_key_override.strip()
            if val and val not in all_keys:
                all_keys.append(val)

        # 3. Database (Manageable via GUI)
        try:
            # Import BackendSettings here to avoid circular imports if any
            from ...core.settings import BackendSettings
            db_key = BackendSettings.get_setting("api_key")
            if db_key and isinstance(db_key, str):
                val = db_key.strip()
                if val and val not in all_keys:
                    all_keys.append(val)
        except Exception:
            # Fallback if DB not ready
            pass

        self.api_keys = all_keys


# Global settings instance
settings = Settings()