"""
Result Cache: Caches job results for performance.
"""

from typing import Dict, Any, Optional, List, Callable, Set, Tuple, Union
from datetime import datetime, timedelta
import uuid
import json
import logging
import time
import threading
import psutil
import os
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ResultCache:
    """Caches job results for performance."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        with self._lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return result
                else:
                    del self.cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        """Set cached result."""
        with self._lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entries
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            self.cache[key] = (value, datetime.utcnow())

    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self.cache.clear()

    def cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        with self._lock:
            now = datetime.utcnow()
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if now - timestamp >= timedelta(seconds=self.ttl_seconds)
            ]
            for key in expired_keys:
                del self.cache[key]