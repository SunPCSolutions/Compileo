"""
Resource Manager: Manages system resources for job execution.
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

from .models import (
    JobStatus, JobPriority, JobType, ResourceType,
    ResourceLimits, JobDependency, JobSchedule, JobMetrics,
    EnhancedExtractionJob
)

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages system resources for job execution."""

    def __init__(self, global_max_jobs: int = 10, per_user_max_jobs: int = 3, get_active_job_counts=None):
        self._lock = threading.Lock()
        self.current_cpu_percent = 0.0
        self.current_memory_mb = 0
        self.active_jobs = 0
        self.api_call_counts: Dict[str, Tuple[int, datetime]] = {}
        self.global_max_concurrent_jobs = global_max_jobs
        self.per_user_max_concurrent_jobs = per_user_max_jobs
        self.user_job_counts: Dict[str, int] = {}
        self._get_active_job_counts = get_active_job_counts  # Callback to get active job counts
        from .queue_processor import QueueProcessor
        self.queue_processor = QueueProcessor(self)

        # Start thread health monitoring
        self._health_monitor_thread = threading.Thread(
            target=self._monitor_thread_health,
            daemon=True,
            name="ThreadHealthMonitor"
        )
        self._health_monitor_thread.start()

    def check_resource_limits(self, limits: ResourceLimits, user_id: Optional[str] = None) -> bool:
        """Check if resource limits allow job execution."""
        try:
            # Get current active job counts
            if self._get_active_job_counts:
                active_counts = self._get_active_job_counts()
                global_active = active_counts.get('global', 0)
                user_active = active_counts.get('user', {}).get(user_id, 0) if user_id else 0
            else:
                # Fallback to local counters (not reliable across processes)
                global_active = self.active_jobs
                user_active = self.user_job_counts.get(user_id, 0) if user_id else 0

            # Check global limit
            # Retrieve latest setting from DB if possible, fallback to init value
            try:
                from ...core.settings import backend_settings
                current_global_limit = backend_settings.get_global_max_concurrent_jobs()
            except ImportError:
                current_global_limit = self.global_max_concurrent_jobs

            if global_active >= current_global_limit:
                logger.debug(f"Global job limit exceeded: {global_active}/{current_global_limit}")
                return False

            # Check per-user limit if user_id provided
            # TODO: Uncomment when multi-user architecture is implemented
            # if user_id and user_active >= self.per_user_max_concurrent_jobs:
            #     logger.debug(f"Per-user job limit exceeded for {user_id}: {user_active}/{self.per_user_max_concurrent_jobs}")
            #     return False

            # Check resource limits if provided
            if limits:
                # Check CPU limit
                if limits.max_cpu_percent and self.current_cpu_percent >= limits.max_cpu_percent:
                    logger.info(f"System CPU limit exceeded: {self.current_cpu_percent}%/{limits.max_cpu_percent}%")
                    return False

                # Check memory limit
                # NOTE: Only enforce memory limit if system usage is very high (>90%)
                # The default limit (4GB) is too low for modern desktops and causes blocking
                mem_percent = psutil.virtual_memory().percent
                if mem_percent > 90:
                    if limits.max_memory_mb and self.current_memory_mb >= limits.max_memory_mb:
                        logger.info(f"System memory limit exceeded: {self.current_memory_mb}MB/{limits.max_memory_mb}MB (Usage: {mem_percent}%)")
                        return False

            logger.debug(f"Resource limits OK - Global: {global_active}/{self.global_max_concurrent_jobs}")
            return True

        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            # On error, allow job to proceed (fail-safe)
            return True

    def allocate_resources(self, limits: ResourceLimits, user_id: Optional[str] = None) -> bool:
        """Allocate resources for job execution."""
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

    def _monitor_thread_health(self) -> None:
        """Monitor health of background threads."""
        while True:
            try:
                # Check QueueProcessor thread health
                if (hasattr(self.queue_processor, '_processor_thread') and
                    self.queue_processor._processor_thread is not None):
                    if not self.queue_processor._processor_thread.is_alive():
                        logger.error("ResourceManager: QueueProcessor thread died, attempting restart")
                        try:
                            self.queue_processor.start_processing()
                            logger.info("ResourceManager: QueueProcessor thread restarted successfully")
                        except Exception as restart_error:
                            logger.error(f"ResourceManager: Failed to restart QueueProcessor thread: {restart_error}")
                    else:
                        logger.debug("ResourceManager: QueueProcessor thread is healthy")
                elif hasattr(self.queue_processor, '_running') and self.queue_processor._running:
                    logger.warning("ResourceManager: QueueProcessor is running but thread is None, attempting restart")
                    try:
                        self.queue_processor.start_processing()
                        logger.info("ResourceManager: QueueProcessor thread restarted from None state")
                    except Exception as restart_error:
                        logger.error(f"ResourceManager: Failed to restart QueueProcessor from None state: {restart_error}")

                # Check other threads if needed
                # Add monitoring for other background threads here

            except Exception as e:
                logger.error(f"ResourceManager: Error in thread health monitoring: {e}")

            time.sleep(30)  # Check every 30 seconds