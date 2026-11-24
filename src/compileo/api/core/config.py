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
    # Legacy X-API-Key removed. Use JWT for authentication.

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
    per_user_max_concurrent_jobs: int = 3  # Per-user concurrent job limit

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

        # Override CORS origins from environment
        cors_origins_env = os.getenv("CORS_ORIGINS", "")
        if cors_origins_env:
            self.cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

        # Override job limits from GUI settings
        backend_settings = BackendSettings()
        self.global_max_concurrent_jobs = backend_settings.get_global_max_concurrent_jobs()
        self.per_user_max_concurrent_jobs = backend_settings.get_per_user_max_concurrent_jobs()


# Global settings instance
settings = Settings()
