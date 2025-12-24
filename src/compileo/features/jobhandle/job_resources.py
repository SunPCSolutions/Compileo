"""
Job Resources Module.
Handles resource management, allocation, and monitoring for job execution.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import logging
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .models import ResourceLimits

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages system resources for job execution."""

    def __init__(self, global_max_jobs: int = 10, per_user_max_jobs: int = 3):
        self._lock = threading.Lock()
        self.current_cpu_percent = 0.0
        self.current_memory_mb = 0
        self.current_gpu_memory_mb = 0
        self.active_jobs = 0
        self.api_call_counts: Dict[str, tuple[int, datetime]] = {}
        self.global_max_concurrent_jobs = global_max_jobs
        self.per_user_max_concurrent_jobs = per_user_max_jobs
        self.user_job_counts: Dict[str, int] = {}
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False

    def check_resource_limits(self, limits: 'ResourceLimits', user_id: Optional[str] = None) -> bool:
        """Check if resource limits allow job execution."""
        with self._lock:
            # Update current system metrics
            self.update_system_metrics()

            # Check global concurrent job limit
            if self.active_jobs >= self.global_max_concurrent_jobs:
                logger.warning(f"Global concurrent job limit exceeded: {self.active_jobs}/{self.global_max_concurrent_jobs}")
                return False

            # Check per-user concurrent job limit
            # TODO: Uncomment when multi-user architecture is implemented
            # if user_id and self.user_job_counts.get(user_id, 0) >= self.per_user_max_concurrent_jobs:
            #     logger.warning(f"Per-user concurrent job limit exceeded for {user_id}: {self.user_job_counts.get(user_id, 0)}/{self.per_user_max_concurrent_jobs}")
            #     return False

            # Check CPU usage
            if self.current_cpu_percent >= limits.max_cpu_percent:
                logger.warning(f"CPU usage too high: {self.current_cpu_percent}% >= {limits.max_cpu_percent}%")
                return False

            # Check memory usage
            if self.current_memory_mb >= limits.max_memory_mb:
                logger.warning(f"Memory usage too high: {self.current_memory_mb}MB >= {limits.max_memory_mb}MB")
                return False

            # Check GPU memory usage if GPU is available
            if self.gpu_available:
                gpu_memory_limit_mb = getattr(limits, 'max_gpu_memory_mb', 8192)  # Default 8GB GPU limit
                if self.current_gpu_memory_mb >= gpu_memory_limit_mb:
                    logger.warning(f"GPU memory usage too high: {self.current_gpu_memory_mb}MB >= {gpu_memory_limit_mb}MB")
                    return False

            # Check API rate limits (simplified - just count per minute)
            api_key = f"user_{user_id}" if user_id else "anonymous"
            if api_key in self.api_call_counts:
                count, reset_time = self.api_call_counts[api_key]
                if datetime.utcnow() < reset_time and count >= limits.max_api_calls_per_minute:
                    logger.warning(f"API rate limit exceeded for {api_key}: {count}/{limits.max_api_calls_per_minute}")
                    return False

            logger.info(f"Resource check passed: CPU={self.current_cpu_percent:.1f}%, Memory={self.current_memory_mb:.0f}MB, Active jobs={self.active_jobs}")
            return True

    def allocate_resources(self, limits: 'ResourceLimits', user_id: Optional[str] = None) -> bool:
        """Allocate resources for job execution."""
        with self._lock:
            logger.info(f"Checking limits in allocate_resources: active_jobs={self.active_jobs}, user_jobs={self.user_job_counts.get(user_id, 0) if user_id else 'N/A'}")
            if not self.check_resource_limits(limits, user_id):
                logger.warning("Resource limit check failed in allocate_resources")
                return False

            self.active_jobs += 1
            if user_id:
                self.user_job_counts[user_id] = self.user_job_counts.get(user_id, 0) + 1
            logger.info(f"Resources allocated: active_jobs={self.active_jobs}, user_jobs={self.user_job_counts.get(user_id, 0) if user_id else 'N/A'}")
            # In a real implementation, you'd track actual CPU/memory usage
            return True

    def release_resources(self, user_id: Optional[str] = None) -> None:
        """Release resources after job completion."""
        with self._lock:
            self.active_jobs = max(0, self.active_jobs - 1)
            if user_id:
                if user_id in self.user_job_counts:
                    self.user_job_counts[user_id] = max(0, self.user_job_counts[user_id] - 1)
                    if self.user_job_counts[user_id] == 0:
                        del self.user_job_counts[user_id]

    def update_system_metrics(self) -> None:
        """Update current system resource usage."""
        with self._lock:
            self.current_cpu_percent = psutil.cpu_percent(interval=1)
            self.current_memory_mb = psutil.virtual_memory().used / (1024 * 1024)

            # Update GPU memory usage if available
            if self.gpu_available and TORCH_AVAILABLE:
                try:
                    self.current_gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                except Exception as e:
                    logger.warning(f"Failed to get GPU memory usage: {e}")
                    self.current_gpu_memory_mb = 0
            else:
                self.current_gpu_memory_mb = 0

    def record_api_call(self, key: str) -> None:
        """Record an API call for rate limiting."""
        with self._lock:
            now = datetime.utcnow()
            reset_time = now + timedelta(minutes=1)

            if key in self.api_call_counts:
                count, existing_reset = self.api_call_counts[key]
                if now > existing_reset:
                    self.api_call_counts[key] = (1, reset_time)
                else:
                    self.api_call_counts[key] = (count + 1, existing_reset)
            else:
                self.api_call_counts[key] = (1, reset_time)