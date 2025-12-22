"""
Enhanced caching system for extraction results with performance optimizations.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading
import time
import logging

logger = logging.getLogger(__name__)


class ResultCacheInterface:
    """Protocol for result caching implementations."""

    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL in seconds."""
        ...

    def delete(self, key: str) -> bool:
        """Delete cached value by key."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of entries removed."""
        ...


class MemoryResultCache(ResultCacheInterface):
    """In-memory result cache implementation with advanced features."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }

        # Background cleanup thread
        self._cleanup_thread = None
        self._cleanup_interval = 300  # 5 minutes
        self._running = False
        self._start_background_cleanup()

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry:
            expires_at = entry.get('expires_at')
            if expires_at and datetime.utcnow() > expires_at:
                del self._cache[key]
                del self._access_times[key]
                self._stats['misses'] += 1
                return None
            self._access_times[key] = datetime.utcnow()
            self._stats['hits'] += 1
            return entry['value']
        self._stats['misses'] += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Evict if at capacity (LRU)
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        expires_at = None
        if ttl or self.default_ttl:
            ttl_value = ttl or self.default_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_value)

        self._cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'size': self._estimate_size(value)
        }
        self._access_times[key] = datetime.utcnow()
        self._stats['sets'] += 1

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()
        self._access_times.clear()
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'sets': 0}

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return

        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
        self._stats['evictions'] += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        if isinstance(value, (int, float, bool)):
            return 28  # Approximate size of Python objects
        elif isinstance(value, str):
            return len(value) * 2  # UTF-8 encoding
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(item) for item in value) + 64
        elif isinstance(value, dict):
            return sum(len(k) * 2 + self._estimate_size(v) for k, v in value.items()) + 240
        else:
            return 100  # Default estimate

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.get('size', 0) for entry in self._cache.values())
        hit_rate = self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) if (self._stats['hits'] + self._stats['misses']) > 0 else 0

        return {
            'entries': len(self._cache),
            'max_size': self.max_size,
            'total_size_bytes': total_size,
            'hit_rate': hit_rate,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'sets': self._stats['sets']
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of entries removed."""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.get('expires_at') and current_time > entry['expires_at']
        ]

        for key in expired_keys:
            del self._cache[key]
            del self._access_times[key]

        return len(expired_keys)

    def _start_background_cleanup(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="CacheCleanupWorker"
        )
        self._cleanup_thread.start()

    def _cleanup_worker(self) -> None:
        """Background cleanup worker."""
        while self._running:
            try:
                removed = self.cleanup_expired()
                if removed > 0:
                    logger.debug(f"Cleaned up {removed} expired cache entries")
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
            time.sleep(self._cleanup_interval)

    def stop_background_cleanup(self) -> None:
        """Stop background cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)


class MultiLevelCache:
    """Multi-level caching with L1 (fast) and L2 (persistent) caches."""

    def __init__(self, l1_cache: ResultCacheInterface, l2_cache: Optional[ResultCacheInterface] = None):
        self.l1_cache = l1_cache  # Fast in-memory cache
        self.l2_cache = l2_cache  # Slower but persistent cache

    def get(self, key: str) -> Optional[Any]:
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Try L2 cache if available
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value is not None:
                # Promote to L1 cache
                self.l1_cache.set(key, value)
                return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Set in both caches
        self.l1_cache.set(key, value, ttl)
        if self.l2_cache:
            self.l2_cache.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        deleted_l1 = self.l1_cache.delete(key)
        deleted_l2 = self.l2_cache.delete(key) if self.l2_cache else False
        return deleted_l1 or deleted_l2

    def clear(self) -> None:
        self.l1_cache.clear()
        if self.l2_cache:
            self.l2_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        stats = {'l1': self.l1_cache.get_stats()}
        if self.l2_cache:
            stats['l2'] = self.l2_cache.get_stats()
        return stats


class CacheManager:
    """Central cache management with multiple cache instances."""

    def __init__(self):
        self.caches: Dict[str, ResultCacheInterface] = {}
        self._default_cache = MemoryResultCache()

    def get_cache(self, name: str) -> ResultCacheInterface:
        """Get or create a named cache."""
        if name not in self.caches:
            self.caches[name] = MemoryResultCache()
        return self.caches[name]

    def set_cache(self, name: str, cache: ResultCacheInterface) -> None:
        """Set a named cache."""
        self.caches[name] = cache

    def get_default_cache(self) -> ResultCacheInterface:
        """Get the default cache."""
        return self._default_cache

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {'default': self._default_cache.get_stats()}
        for name, cache in self.caches.items():
            stats[name] = cache.get_stats()
        return stats

    def cleanup_all(self) -> Dict[str, int]:
        """Clean up all caches. Returns cleanup counts."""
        counts = {'default': self._default_cache.cleanup_expired()}
        for name, cache in self.caches.items():
            if hasattr(cache, 'cleanup_expired'):
                counts[name] = cache.cleanup_expired()
        return counts


# Global cache manager instance
cache_manager = CacheManager()