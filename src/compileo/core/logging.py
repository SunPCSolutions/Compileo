"""
Centralized logging utility for Compileo.
Provides a unified logger and configuration that respects the global Log Level setting.
"""

import logging
import sys
from typing import Optional
from .settings import BackendSettings, LogLevel

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)

def setup_logging(level: Optional[LogLevel] = None):
    """
    Configure logging based on the provided level or the global setting.
    
    Args:
        level: LogLevel to use. If None, reads from BackendSettings.
    """
    if level is None:
        level = BackendSettings.get_log_level()
    
    # Map LogLevel to logging module levels
    log_mapping = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.NONE: logging.CRITICAL + 1
    }
    
    target_level = log_mapping.get(level, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=target_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    # Force level on all active loggers to ensure consistency across libraries
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(target_level)
    
    # Ensure compileo logger is set correctly
    compileo_logger = logging.getLogger("compileo")
    compileo_logger.setLevel(target_level)
    
    # Special handling for uvicorn and rq to ensure they respect the global setting
    for noisy_logger in ["uvicorn", "uvicorn.error", "uvicorn.access", "rq.worker"]:
        logger_obj = logging.getLogger(noisy_logger)
        logger_obj.setLevel(target_level)
        
        # In RQ 2.x+, the worker will only skip its default log setup if handlers are present.
        # Adding a NullHandler ensures our centralized configuration is used without RQ overrides.
        if noisy_logger == "rq.worker" and not logger_obj.handlers:
            logger_obj.addHandler(logging.NullHandler())
    
    # Silence extremely noisy third-party loggers unless in DEBUG mode
    if level != LogLevel.DEBUG:
        for noisy_lib in ["urllib3", "requests", "google", "ollama", "asyncio"]:
            logging.getLogger(noisy_lib).setLevel(logging.ERROR)
            
    if level != LogLevel.NONE:
        compileo_logger.debug(f"Logging initialized at level: {level.value}")

# Global logger for general use
logger = logging.getLogger("compileo")
