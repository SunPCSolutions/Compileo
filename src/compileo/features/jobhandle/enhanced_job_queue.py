"""Enhanced job queue system for background extraction processing."""

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
from ...core.logging import get_logger

from .models import (
    JobStatus, JobPriority, JobType, ResourceType,
    ResourceLimits, JobDependency, JobSchedule, JobMetrics,
    EnhancedExtractionJob
)
from .worker_functions import redis_job_wrapper
from .queue_processor import QueueProcessor
from .resource_manager import ResourceManager
from .job_scheduler import JobScheduler
from .result_cache import ResultCache


logger = get_logger(__name__)












class EnhancedJobQueueInterface:
    """Interface for enhanced job queue implementations."""

    def enqueue_job(self, job: EnhancedExtractionJob) -> str:
        """Enqueue a job for processing."""
        ...

    def get_job_status(self, job_id: str) -> Optional[EnhancedExtractionJob]:
        """Get job status by ID."""
        ...

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job by ID."""
        ...

    def get_pending_jobs(self, limit: int = 50) -> List[EnhancedExtractionJob]:
        """Get pending jobs."""
        ...

    def get_running_jobs(self, limit: int = 50) -> List[EnhancedExtractionJob]:
        """Get running jobs."""
        ...


class EnhancedRedisJobQueue(EnhancedJobQueueInterface):
    """Enhanced Redis-based job queue with advanced features."""

    def __init__(
        self,
        redis_url: str = 'redis://localhost:6379/0',
        queue_name: str = 'extraction_jobs',
        db_connection=None,
        global_max_jobs: int = 10,
        per_user_max_jobs: int = 3,
        enable_background_monitoring: bool = True
    ):
        from redis import Redis
        from rq import Queue

        try:
            self.redis = Redis.from_url(redis_url)
            # Test the connection
            self.redis.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
            raise

        self.queue_name = queue_name
        self.queue = Queue(queue_name, connection=self.redis, default_timeout=259200)  # 72 hours default
        self.db_connection = db_connection

        # Enhanced components
        # Create callback to get active job counts
        def get_active_job_counts():
            """Get current active job counts from Redis."""
            try:
                # Count running jobs from Redis
                cursor = 0
                global_active = 0
                user_active = {}

                while True:
                    cursor, keys = self.redis.scan(cursor, match="compileo:job:*", count=100)
                    for key in keys:
                        try:
                            key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                            if key_str.endswith(':processing'):
                                continue

                            job_data = self.redis.get(key)
                            if job_data:
                                job = EnhancedExtractionJob.from_dict(json.loads(job_data))
                                if job.status == JobStatus.RUNNING:
                                    global_active += 1
                                    if job.user_id:
                                        user_active[job.user_id] = user_active.get(job.user_id, 0) + 1
                        except Exception as e:
                            logger.warning(f"Error counting active job {key}: {e}")
                            continue

                    if cursor == 0:
                        break

                return {'global': global_active, 'user': user_active}
            except Exception as e:
                logger.error(f"Error getting active job counts: {e}")
                return {'global': 0, 'user': {}}

        self.resource_manager = ResourceManager(global_max_jobs, per_user_max_jobs, get_active_job_counts)
        self.scheduler = JobScheduler()
        self.result_cache = ResultCache()

        # Job storage and tracking
        self._job_storage: Dict[str, EnhancedExtractionJob] = {}
        self._completed_jobs: Set[str] = set()

        # Statistics caching for performance
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._stats_cache_time: Optional[datetime] = None
        self._stats_cache_ttl = 30  # Cache stats for 30 seconds

        # Register with queue processor
        self.resource_manager.queue_processor.register_queue(self)

        if enable_background_monitoring:
            # Start background threads
            self._start_background_threads()

            # Perform initial cleanup on startup
            logger.info("Performing initial job cleanup on startup...")
            try:
                # CRITICAL: Always cleanup stale worker registrations for THIS PID before starting
                # This prevents "active worker already exists" errors when Docker restarts the process
                from rq import Worker
                stale_workers = Worker.all(connection=self.redis)
                current_pid = os.getpid()
                for worker in stale_workers:
                    if str(current_pid) in worker.name:
                        try:
                            # Use JSON logging as per AGENTS.md
                            log_msg = {
                                "event": "stale_worker_cleanup",
                                "worker_name": worker.name,
                                "pid": current_pid,
                                "message": "Cleaning up stale worker registration on startup"
                            }
                            logger.info(f"DEBUG: {json.dumps(log_msg)}")
                            self.redis.delete(f"rq:worker:{worker.name}")
                        except Exception as e:
                            logger.warning(f"Failed to cleanup self-stale worker {worker.name}: {e}")
        
                # Cleanup stuck jobs immediately on startup
                stuck_cleaned = self._cleanup_stuck_running_jobs()
                if stuck_cleaned > 0:
                    logger.info(f"Initial cleanup: fixed {stuck_cleaned} stuck running jobs")

                # More aggressive initial cleanup
                initial_cleaned = self._cleanup_old_jobs_from_redis(hours_old=1)  # Clean jobs older than 1 hour
                if initial_cleaned > 0:
                    logger.info(f"Initial cleanup: removed {initial_cleaned} old jobs")

                # Clean RQ registries immediately
                rq_cleaned = self._cleanup_all_rq_registries()
                if rq_cleaned > 0:
                    logger.info(f"Initial RQ registry cleanup: removed {rq_cleaned} jobs")

                # Force stats cache invalidation
                self._invalidate_stats_cache()

            except Exception as e:
                logger.warning(f"Initial cleanup failed: {e}")
        else:
            logger.info("Background monitoring and cleanup threads disabled for this instance")

    def _start_background_threads(self) -> None:
        """Start background monitoring threads."""
        # Resource monitoring thread
        threading.Thread(target=self._resource_monitor_loop, daemon=True, name="ResourceMonitor").start()

        # Cache cleanup thread
        threading.Thread(target=self._cache_cleanup_loop, daemon=True, name="CacheCleanup").start()

        # Job cleanup thread
        threading.Thread(target=self._job_cleanup_loop, daemon=True, name="JobCleanup").start()

        # Worker health check thread
        threading.Thread(target=self._worker_health_check_loop, daemon=True, name="WorkerHealth").start()

        # Scheduler
        self.scheduler.start_scheduler()

        # Queue processor
        self.resource_manager.queue_processor.start_processing()

    def _resource_monitor_loop(self) -> None:
        """Monitor system resources."""
        while True:
            try:
                self.resource_manager.update_system_metrics()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                time.sleep(60)

    def _cache_cleanup_loop(self) -> None:
        """Clean up expired cache entries."""
        while True:
            try:
                self.result_cache.cleanup_expired()
                time.sleep(300)  # Clean up every 5 minutes
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(600)

    def _worker_health_check_loop(self) -> None:
        """Monitor worker health and clean up stale worker registrations."""
        while True:
            try:
                # Check for stale workers (workers that haven't sent heartbeat in 5 minutes)
                from rq import Worker
                workers = Worker.all(connection=self.redis)
                current_time = datetime.utcnow()
                stale_threshold = timedelta(minutes=5)

                stale_workers = []
                for worker in workers:
                    if worker.last_heartbeat:
                        # Handle both offset-naive and offset-aware datetimes
                        if worker.last_heartbeat.tzinfo is not None:
                            # last_heartbeat is timezone-aware, make current_time aware too
                            from datetime import timezone
                            current_time_aware = current_time.replace(tzinfo=timezone.utc)
                            time_since_heartbeat = current_time_aware - worker.last_heartbeat
                        else:
                            # last_heartbeat is naive, keep current_time naive
                            time_since_heartbeat = current_time - worker.last_heartbeat

                        if time_since_heartbeat > stale_threshold:
                            stale_workers.append(worker)

                # Clean up stale worker registrations
                for worker in stale_workers:
                    try:
                        # Remove worker from Redis
                        worker_key = f"rq:worker:{worker.name}"
                        self.redis.delete(worker_key)
                        logger.info(f"Cleaned up stale worker: {worker.name} (last heartbeat: {worker.last_heartbeat})")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup stale worker {worker.name}: {e}")

                if stale_workers:
                    logger.info(f"Worker health check: cleaned up {len(stale_workers)} stale workers")

                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Worker health check error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _job_cleanup_loop(self) -> None:
        """Clean up old completed/failed/cancelled jobs from Redis and RQ registries."""
        while True:
            try:
                total_cleaned = 0

                # Clean up RQ failed jobs more aggressively (every 10 minutes)
                rq_failed_cleaned = self._cleanup_rq_failed_jobs()
                if rq_failed_cleaned > 0:
                    logger.info(f"Periodic RQ failed job cleanup: removed {rq_failed_cleaned} failed jobs")
                    total_cleaned += rq_failed_cleaned

                # Clean up RQ finished jobs (keep only last 24 hours)
                rq_finished_cleaned = self._cleanup_rq_finished_jobs(hours_old=24)
                if rq_finished_cleaned > 0:
                    logger.info(f"Periodic RQ finished job cleanup: removed {rq_finished_cleaned} finished jobs")
                    total_cleaned += rq_finished_cleaned

                # Clean up custom jobs more aggressively (every 2 hours instead of 7 days)
                cleaned_count = self._cleanup_old_jobs_from_redis(hours_old=2)  # Changed from days_old=7 to hours_old=2
                if cleaned_count > 0:
                    logger.info(f"Periodic custom job cleanup: removed {cleaned_count} old jobs")
                    total_cleaned += cleaned_count

                # Clean up stuck running jobs (every cycle)
                stuck_cleaned = self._cleanup_stuck_running_jobs()
                if stuck_cleaned > 0:
                    logger.info(f"Periodic stuck job cleanup: fixed {stuck_cleaned} stuck jobs")
                    total_cleaned += stuck_cleaned

                # Clean up orphaned processing locks (older than 10 minutes)
                lock_cleaned = self._cleanup_orphaned_processing_locks(minutes_old=10)
                if lock_cleaned > 0:
                    logger.info(f"Periodic processing lock cleanup: removed {lock_cleaned} orphaned locks")
                    total_cleaned += lock_cleaned

                if total_cleaned > 0:
                    logger.info(f"Total cleanup cycle: removed {total_cleaned} items")

                time.sleep(600)  # Clean up every 10 minutes (more frequent)
            except Exception as e:
                logger.error(f"Job cleanup error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _cleanup_stuck_running_jobs(self) -> int:
        """Clean up jobs that are stuck in RUNNING state but have no active worker."""
        cleaned_count = 0
        from rq.job import Job
        from rq.exceptions import NoSuchJobError

        try:
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match="compileo:job:*", count=100)
                for key_bytes in keys:
                    try:
                        key_str = key_bytes.decode('utf-8') if isinstance(key_bytes, bytes) else str(key_bytes)
                        if key_str.endswith(':processing'):
                            continue

                        job_data = self.redis.get(key_bytes)
                        if job_data:
                            job = EnhancedExtractionJob.from_dict(json.loads(job_data))
                            
                            if job.status == JobStatus.RUNNING:
                                # Check if processing lock exists
                                lock_key = f"compileo:job:{job.job_id}:processing"
                                if self.redis.exists(lock_key):
                                    continue  # Job has active lock, it's fine
                                
                                # Check grace period for newly started jobs (to allow time for enqueueing/worker pickup)
                                # If job started less than 60 seconds ago, assume it's initializing
                                if job.started_at:
                                    # Handle both offset-naive and offset-aware datetimes
                                    current_time = datetime.utcnow()
                                    started_at = job.started_at
                                    
                                    if started_at.tzinfo is not None:
                                        from datetime import timezone
                                        current_time = current_time.replace(tzinfo=timezone.utc)
                                    
                                    time_since_start = (current_time - started_at).total_seconds()
                                    if time_since_start < 60:
                                        logger.debug(f"Job {job.job_id} is RUNNING but no lock yet - within 60s grace period ({time_since_start}s). Skipping cleanup.")
                                        continue

                                # No lock - check RQ status
                                is_stuck = False
                                try:
                                    rq_job = Job.fetch(job.job_id, connection=self.redis)
                                    rq_status = rq_job.get_status()
                                    if rq_status in ['queued', 'started', 'deferred', 'scheduled']:
                                        continue  # Job is waiting in queue or starting, it's fine
                                    
                                    # If RQ job is finished/failed but our status is RUNNING -> Stuck
                                    logger.warning(f"Job {job.job_id} is RUNNING but RQ status is {rq_status}. Marking as FAILED.")
                                    is_stuck = True
                                    
                                except (NoSuchJobError, Exception):
                                    # RQ job missing -> Stuck (Lost)
                                    logger.warning(f"Job {job.job_id} is RUNNING but missing from RQ. Marking as FAILED.")
                                    is_stuck = True
                                
                                if is_stuck:
                                    # Mark as FAILED
                                    job.status = JobStatus.FAILED
                                    job.completed_at = datetime.utcnow()
                                    job.error = "Stuck in RUNNING state detected - no active worker or RQ job"
                                    
                                    # Update Redis
                                    self.redis.set(key_bytes, json.dumps(job.to_dict()))
                                    cleaned_count += 1
                                    logger.info(f"Fixed stuck job {job.job_id}: Set status to FAILED")

                    except Exception as e:
                        logger.warning(f"Error checking job {key_bytes}: {e}")
                        continue

                if cursor == 0:
                    break

            if cleaned_count > 0:
                self._invalidate_stats_cache()
                
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup stuck jobs: {e}")
            return 0

    def _cleanup_old_jobs_from_redis(self, hours_old: int = 48) -> int:
        """Clean up old jobs from Redis (internal method)."""
        cutoff_date = datetime.utcnow() - timedelta(hours=hours_old)
        cleaned_count = 0

        try:
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match="compileo:job:*", count=100)
                for key_bytes in keys:
                    try:
                        # Skip processing lock keys (they contain worker info, not job data)
                        key_str = key_bytes.decode('utf-8') if isinstance(key_bytes, bytes) else str(key_bytes)
                        if key_str.endswith(':processing'):
                            continue

                        job_data = self.redis.get(key_bytes)
                        if job_data:
                            job = EnhancedExtractionJob.from_dict(json.loads(job_data))

                            # Clean up jobs that are completed, failed, or cancelled and older than cutoff
                            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                                job.completed_at and job.completed_at < cutoff_date):

                                # Remove from Redis
                                self.redis.delete(key_bytes)
                                cleaned_count += 1

                                # Remove from completed jobs set if present
                                if hasattr(self, '_completed_jobs') and job.job_id in self._completed_jobs:
                                    self._completed_jobs.discard(job.job_id)

                    except Exception as e:
                        logger.warning(f"Failed to process job key {key_bytes}: {e}")
                        continue

                if cursor == 0:
                    break

            # Invalidate stats cache after cleanup
            self._invalidate_stats_cache()

            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup old jobs from Redis: {e}")
            return 0

    def _cleanup_rq_failed_jobs(self) -> int:
        """Clean up RQ failed jobs from the failed job registry."""
        try:
            failed_jobs = self.queue.failed_job_registry.get_job_ids()
            cleaned_count = 0

            for job_id in failed_jobs:
                try:
                    job = self.queue.fetch_job(job_id)
                    if job:
                        # Remove from failed registry and delete job
                        self.queue.failed_job_registry.remove(job_id)
                        job.delete()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup RQ failed job {job_id}: {e}")

            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup RQ failed jobs: {e}")
            return 0

    def _cleanup_all_rq_registries(self) -> int:
        """Clean up all RQ job registries aggressively on startup."""
        total_cleaned = 0

        try:
            # Clean failed jobs
            failed_cleaned = self._cleanup_rq_failed_jobs()
            total_cleaned += failed_cleaned

            # Clean finished jobs (older than 1 hour on startup)
            finished_cleaned = self._cleanup_rq_finished_jobs(hours_old=1)
            total_cleaned += finished_cleaned

            # Clean canceled jobs
            try:
                canceled_registry = self.queue.canceled_job_registry
                canceled_jobs = canceled_registry.get_job_ids()
                for job_id in canceled_jobs:
                    try:
                        job = self.queue.fetch_job(job_id)
                        if job:
                            canceled_registry.remove(job_id, delete_job=True)
                            total_cleaned += 1
                    except Exception as e:
                        logger.warning(f"Failed to cleanup canceled job {job_id}: {e}")
            except Exception as e:
                logger.warning(f"Failed to access canceled job registry: {e}")

            # Clean deferred jobs that are stuck
            try:
                deferred_registry = self.queue.deferred_job_registry
                deferred_jobs = deferred_registry.get_job_ids()
                current_time = datetime.utcnow()

                for job_id in deferred_jobs:
                    try:
                        job = self.queue.fetch_job(job_id)
                        if job and job.created_at:
                            # Handle both offset-naive and offset-aware datetimes
                            if job.created_at.tzinfo is not None:
                                # created_at is timezone-aware, make current_time aware too
                                from datetime import timezone
                                current_time_aware = current_time.replace(tzinfo=timezone.utc)
                                time_diff = (current_time_aware - job.created_at).total_seconds()
                            else:
                                # created_at is naive, keep current_time naive
                                time_diff = (current_time - job.created_at).total_seconds()

                            # If deferred for more than 1 hour, likely stuck
                            if time_diff > 3600:
                                deferred_registry.remove(job_id, delete_job=True)
                                total_cleaned += 1
                                logger.debug(f"Cleaned stuck deferred job: {job_id}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup deferred job {job_id}: {e}")
            except Exception as e:
                logger.warning(f"Failed to access deferred job registry: {e}")

            return total_cleaned

        except Exception as e:
            logger.error(f"Failed to cleanup RQ registries: {e}")
            return 0

    def _cleanup_rq_finished_jobs(self, hours_old: int = 24) -> int:
        """Clean up RQ finished jobs older than specified hours."""
        try:
            finished_registry = self.queue.finished_job_registry
            finished_jobs = finished_registry.get_job_ids()
            cleaned_count = 0
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_old)

            for job_id in finished_jobs:
                try:
                    job = self.queue.fetch_job(job_id)
                    if job and job.ended_at:
                        # Handle both offset-naive and offset-aware datetimes
                        if job.ended_at.tzinfo is not None:
                            # ended_at is timezone-aware, make cutoff_time aware too
                            from datetime import timezone
                            cutoff_time_aware = cutoff_time.replace(tzinfo=timezone.utc)
                            should_cleanup = job.ended_at < cutoff_time_aware
                        else:
                            # ended_at is naive, keep cutoff_time naive
                            should_cleanup = job.ended_at < cutoff_time

                        if should_cleanup:
                            # Remove from finished registry and delete job
                            finished_registry.remove(job_id)
                            job.delete()
                            cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup RQ finished job {job_id}: {e}")

            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup RQ finished jobs: {e}")
            return 0

    def _cleanup_orphaned_processing_locks(self, minutes_old: int = 10) -> int:
        """Clean up orphaned processing locks older than specified minutes."""
        try:
            cursor = 0
            cleaned_count = 0
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes_old)

            while True:
                cursor, keys = self.redis.scan(cursor, match="compileo:job:*:processing", count=100)
                for key_bytes in keys:
                    try:
                        key = key_bytes.decode('utf-8') if isinstance(key_bytes, bytes) else str(key_bytes)
                        lock_value = self.redis.get(key_bytes)

                        if lock_value:
                            # Parse lock value to check timestamp
                            lock_str = lock_value.decode('utf-8') if isinstance(lock_value, bytes) else str(lock_value)
                            # Format: worker_{pid}_{timestamp}
                            parts = lock_str.split('_')
                            if len(parts) >= 3:
                                try:
                                    timestamp_str = '_'.join(parts[2:])  # Handle timestamp with underscores
                                    lock_time = datetime.fromisoformat(timestamp_str.replace('_', 'T').split('+')[0])

                                    if lock_time < cutoff_time:
                                        self.redis.delete(key_bytes)
                                        cleaned_count += 1
                                        logger.debug(f"Cleaned up orphaned processing lock: {key}")
                                except (ValueError, IndexError) as e:
                                    # Invalid timestamp format, clean it up
                                    self.redis.delete(key_bytes)
                                    cleaned_count += 1
                                    logger.debug(f"Cleaned up malformed processing lock: {key}")

                    except Exception as e:
                        logger.warning(f"Failed to process processing lock {key_bytes}: {e}")

                if cursor == 0:
                    break

            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned processing locks: {e}")
            return 0

    def enqueue_job(self, job: EnhancedExtractionJob) -> str:
        """Enqueue job for processing."""
        from datetime import datetime
        logger.info(f"Enqueueing job {job.job_id} ({job.job_type})")

        self.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict()))

        if job.schedule.scheduled_time and job.schedule.scheduled_time > datetime.utcnow():
            self.scheduler.schedule_job(job)
        else:
            if self.resource_manager.check_resource_limits(job.resource_limits, job.user_id):
                scheduling_lock_key = f"compileo:job:{job.job_id}:scheduling_lock"
                lock_acquired = self.redis.set(scheduling_lock_key, "locked", ex=10, nx=True)
                
                if not lock_acquired:
                    return job.job_id

                try:
                    current_job_data = self.redis.get(f"compileo:job:{job.job_id}")
                    if current_job_data:
                        current_status = json.loads(current_job_data).get("status")
                        if current_status != JobStatus.PENDING.value:
                            return job.job_id

                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.utcnow()
                    self.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict()))
                finally:
                    self.redis.delete(scheduling_lock_key)

                rq_priority = self._priority_to_rq(job.priority)
                try:
                    from .worker_wrapper import process_job
                    rq_job = self.queue.enqueue(
                        'src.compileo.features.jobhandle.worker_wrapper.process_job',
                        job.job_id,
                        priority=rq_priority,
                        timeout=job.schedule.timeout_seconds
                    )
                    logger.info(f"Job {job.job_id} enqueued (RQ ID: {rq_job.id})")
                except Exception as e:
                    logger.error(f"Failed to enqueue job {job.job_id}: {e}")
                    job.status = JobStatus.PENDING
                    job.started_at = None
                    self.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict()))
                    raise
            else:
                logger.info(f"Job {job.job_id} queued (pending resources)")

        self._invalidate_stats_cache()
        return job.job_id



    def _execute_job(self, job_type: str, parameters: Dict[str, Any], job: EnhancedExtractionJob) -> Any:
        """Execute a job based on its type with progress tracking."""
        from .progress_context import ProgressContext
        from .job_executors import _execute_chunk_documents, _execute_parse_documents, _execute_taxonomy_processing, _execute_dataset_generation
        # Simple debug print to check if method is called
        logger.debug(f"_execute_job called for job {job.job_id} with type {job_type}")

        # Create progress context for this job
        progress_context = ProgressContext(job, self.redis)

        # Normalize job type to string for comparison
        if hasattr(job_type, 'value'):
            job_type_str = job_type.value
        else:
            job_type_str = str(job_type)

        logger.debug(f"_execute_job called with job_type={job_type}, job_type_str={job_type_str}, job_id={job.job_id}")

        try:
            # Debug: Log job execution path
            debug_context = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "_execute_job",
                "message": "Starting job execution",
                "context": {
                    "job_type": str(job_type),
                    "job_type_str": job_type_str,
                    "operation": parameters.get("operation"),
                    "job_id": job.job_id
                }
            }
            logger.debug(json.dumps(debug_context))

            if job_type_str in ['extraction', 'document_processing']:
                # Handle document parsing and chunking operations
                operation = parameters.get("operation")
                logger.debug(f"Processing document processing job with operation={operation}, job_id={job.job_id}")
                debug_context = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "component": "_execute_job",
                    "message": "Processing document processing job",
                    "context": {
                        "operation": operation,
                        "job_id": job.job_id,
                        "all_parameters": list(parameters.keys())
                    }
                }
                logger.debug(json.dumps(debug_context))

                if operation == "parse_documents":
                    return _execute_parse_documents(parameters, progress_context, self.db_connection)
                elif operation == "chunk_documents":
                    debug_context = {
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "component": "_execute_job",
                        "message": "Calling _execute_chunk_documents",
                        "context": {"job_id": job.job_id}
                    }
                    logger.debug(json.dumps(debug_context))
                    return _execute_chunk_documents(parameters, progress_context, self.db_connection)
                elif operation == "extraction":
                    return self._execute_extraction_job(parameters, progress_context)
                else:
                    raise ValueError(f"Unknown document processing operation: {operation}")

            elif job_type_str == 'batch_extraction':
                # Simulate batch extraction job
                progress_context.set_total_steps(20)
                for i in range(20):
                    time.sleep(0.05)
                    progress_context.update_progress(message=f"Processing batch item {i + 1}")
                return {"status": "completed", "processed_batches": 20, **parameters}

            elif job_type_str == 'taxonomy_processing':
                return _execute_taxonomy_processing(parameters, progress_context)

            elif job_type_str == 'dataset_generation':
                return _execute_dataset_generation(parameters, progress_context)

            elif job_type_str == 'benchmarking':
                from .job_executors import execute_benchmark_job
                return execute_benchmark_job(parameters, progress_context)

            elif job_type_str == 'cleanup':
                # Simulate cleanup job
                progress_context.set_total_steps(1)
                time.sleep(0.5)
                progress_context.update_progress(message="Cleanup completed")
                return {"status": "completed", "cleaned_items": 100, **parameters}

            elif job_type_str == 'maintenance':
                # Simulate maintenance job
                progress_context.set_total_steps(1)
                time.sleep(1.0)
                progress_context.update_progress(message="Maintenance completed")
                return {"status": "completed", "maintenance_tasks": 5, **parameters}

            else:
                raise ValueError(f"Unknown job type: {job_type}")

        except Exception as e:
            logger.error(f"Job execution failed: {e}")
            raise




    def _execute_taxonomy_processing(self, parameters: Dict[str, Any], progress_context) -> Dict[str, Any]:
        """Execute taxonomy processing with progress tracking."""
        from ...core.settings import backend_settings

        # Taxonomy parameters
        chunks = parameters.get("chunks", [])
        domain = parameters.get("domain", "general")
        depth = parameters.get("depth", 3)
        sample_size = parameters.get("sample_size")
        category_limits = parameters.get("category_limits")
        specificity_level = parameters.get("specificity_level", 1)
        generator_name = parameters.get("generator", "ollama")

        # Initialize taxonomy generator
        # Propagate model from parameters if available, otherwise get from database settings
        model = parameters.get("model")

        if generator_name == "gemini":
            from ...features.taxonomy.gemini_generator import GeminiTaxonomyGenerator
            api_key = backend_settings.get_setting("gemini_api_key")
            if not model:
                model = backend_settings.get_taxonomy_gemini_model()
            generator = GeminiTaxonomyGenerator(api_key=api_key, model=model)
            taxonomy_options = {}
        elif generator_name == "grok":
            from ...features.taxonomy.grok_generator import GrokTaxonomyGenerator
            api_key = backend_settings.get_setting("grok_api_key")
            if not model:
                model = backend_settings.get_taxonomy_grok_model()
            generator = GrokTaxonomyGenerator(grok_api_key=api_key, model=model)
            taxonomy_options = {}
        elif generator_name == "openai":
            from ...features.taxonomy.openai_generator import OpenAITaxonomyGenerator
            api_key = backend_settings.get_openai_api_key()
            if not model:
                model = backend_settings.get_taxonomy_openai_model()
            generator = OpenAITaxonomyGenerator(api_key=api_key, model=model)
            taxonomy_options = {}
        else:
            from ...features.taxonomy.ollama_generator import OllamaTaxonomyGenerator
            if not model:
                model = backend_settings.get_taxonomy_ollama_model()
            generator = OllamaTaxonomyGenerator(model=model)
            # Retrieve taxonomy Ollama parameters
            taxonomy_options = {
                "temperature": backend_settings.get_taxonomy_ollama_temperature(),
                "repeat_penalty": backend_settings.get_taxonomy_ollama_repeat_penalty(),
                "top_p": backend_settings.get_taxonomy_ollama_top_p(),
                "top_k": backend_settings.get_taxonomy_ollama_top_k(),
                "num_predict": backend_settings.get_taxonomy_ollama_num_predict(),
                "seed": backend_settings.get_taxonomy_ollama_seed()
            }

        # Generate taxonomy
        progress_context.update_progress(message="Generating taxonomy")
        result = generator.generate_taxonomy(
            chunks=chunks,
            domain=domain,
            depth=depth,
            batch_size=sample_size,
            category_limits=category_limits,
            specificity_level=specificity_level,
            options=taxonomy_options,
            processing_mode=processing_mode
        )

        progress_context.update_progress(message="Taxonomy generation completed")

        return {
            "status": "completed",
            "taxonomy_result": result,
            **parameters
        }

    def _execute_dataset_generation(self, parameters: Dict[str, Any], progress_context) -> Dict[str, Any]:
        """Execute dataset generation with progress tracking."""
        from ...core.settings import backend_settings
        from ...features.datasetgen.generator import DatasetGenerator
        from ...features.datasetgen.prompt_builder import PromptBuilder
        from ...features.datasetgen.output_formatter import OutputFormatter
        from ...features.datasetgen.llm_interaction import LLMInteraction

        # Retrieve generation Ollama parameters
        generation_options = {
            "temperature": backend_settings.get_generation_ollama_temperature(),
            "repeat_penalty": backend_settings.get_generation_ollama_repeat_penalty(),
            "top_p": backend_settings.get_generation_ollama_top_p(),
            "top_k": backend_settings.get_generation_ollama_top_k(),
            "num_predict": backend_settings.get_generation_ollama_num_predict(),
            "seed": backend_settings.get_generation_ollama_seed()
        }

        # Extract parameters
        project_id = parameters.get("project_id")
        chunks = parameters.get("chunks")
        prompt_name = parameters.get("prompt_name", "qa_pairs")
        format_type = parameters.get("format_type", "json")
        concurrency = parameters.get("concurrency", 1)
        taxonomy_project = parameters.get("taxonomy_project")
        taxonomy_name = parameters.get("taxonomy_name")
        dataset_name = parameters.get("dataset_name")
        enable_versioning = parameters.get("enable_versioning", False)
        datasets_per_chunk = parameters.get("datasets_per_chunk", 3)
        prefer_extracted = parameters.get("prefer_extracted", True)

        # Get LLM provider and model from parameters
        # Propagate from parameters if available, otherwise get from database settings
        llm_provider = parameters.get("classification_provider") or parameters.get("classification_model", "ollama")
        generation_model = parameters.get("generation_model")
        
        # Get API key and specific model from settings based on provider
        api_key = None
        if llm_provider == "gemini":
            api_key = backend_settings.get_setting("gemini_api_key")
            if not generation_model:
                generation_model = backend_settings.get_generation_gemini_model()
        elif llm_provider == "grok":
            api_key = backend_settings.get_setting("grok_api_key")
            if not generation_model:
                generation_model = backend_settings.get_generation_grok_model()
        elif llm_provider == "openai":
            api_key = backend_settings.get_openai_api_key()
            if not generation_model:
                generation_model = backend_settings.get_generation_openai_model()
        elif llm_provider == "ollama":
            if not generation_model:
                generation_model = backend_settings.get_generation_ollama_model()

        # Initialize components
        prompt_builder = PromptBuilder()
        output_formatter = OutputFormatter()
        llm_interaction = LLMInteraction(llm_provider=llm_provider, api_key=api_key, model=generation_model)

        # Create dataset generator
        generator = DatasetGenerator(
            prompt_builder=prompt_builder,
            output_formatter=output_formatter,
            llm_interaction=llm_interaction,
            document_repository=None,  # Not needed for basic generation
            taxonomy_loader=None  # Not needed for basic generation
        )

        # Generate dataset
        progress_context.update_progress(message="Generating dataset")
        result = generator.generate_dataset(
            project_id=project_id,
            chunks=chunks,
            prompt_name=prompt_name,
            format_type=format_type,
            concurrency=concurrency,
            taxonomy_project=taxonomy_project,
            taxonomy_name=taxonomy_name,
            dataset_name=dataset_name,
            enable_versioning=enable_versioning,
            datasets_per_chunk=datasets_per_chunk,
            prefer_extracted=prefer_extracted,
            options=generation_options
        )

        progress_context.update_progress(message="Dataset generation completed")

        return {
            "status": "completed",
            "dataset_result": result,
            **parameters
        }

    def _execute_extraction_job(self, parameters: Dict[str, Any], progress_context) -> Dict[str, Any]:
        """Execute taxonomy extraction with progress tracking."""
        from ...core.settings import backend_settings
        from ...features.extraction.taxonomy_extractor import TaxonomyExtractor
        from ...storage.src.project.database_repositories import TaxonomyRepository, ChunkRepository, ExtractionJobRepository, ExtractionResultRepository
        from ...features.extraction.filesystem_storage import HybridStorageManager, FilesystemStorageManager, ExtractionFileManager

        # Extract parameters
        taxonomy_id = parameters.get('taxonomy_id')
        selected_categories = parameters.get('selected_categories', [])
        job_parameters = parameters.get('parameters', {})
        initial_classifier = parameters.get('initial_classifier', 'grok')
        enable_validation_stage = parameters.get('enable_validation_stage', False)
        validation_classifier = parameters.get('validation_classifier')
        extraction_type = parameters.get('extraction_type', 'ner')
        extraction_mode = parameters.get('extraction_mode', 'contextual')

        # Check what parameters are available
        logger.debug(f"parameters keys = {list(parameters.keys())}")
        logger.debug(f"extraction_mode = {parameters.get('extraction_mode', 'NOT_FOUND')}")
        logger.debug(f"full parameters = {json.dumps(parameters, default=str)}")
        job_id = progress_context.job.job_id

        # Get taxonomy information from new taxonomies table
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT project_id, structure FROM taxonomies WHERE id = ?", (taxonomy_id,))
            tax_record = cursor.fetchone()
            if not tax_record:
                raise ValueError(f"Taxonomy {taxonomy_id} not found")

            project_id, taxonomy_data_json = tax_record
        except Exception as e:
            raise ValueError(f"Failed to query taxonomy {taxonomy_id}: {e}")

        # Get project name
        try:
            cursor.execute("SELECT name FROM projects WHERE id = ?", (project_id,))
            project_record = cursor.fetchone()
            if not project_record:
                raise ValueError(f"Project with ID {project_id} not found")
            project_name = project_record[0]  # Extract name from tuple
        except Exception as e:
            raise ValueError(f"Failed to query project {project_id}: {e}")

        progress_context.update_progress(message="Initializing taxonomy extractor")

        # Create extractor
        grok_key = backend_settings.get_setting("grok_api_key")
        gemini_key = backend_settings.get_setting("gemini_api_key")
        openai_key = backend_settings.get_openai_api_key()

        taxonomy_repo = TaxonomyRepository(self.db_connection)
        chunk_repo = ChunkRepository(self.db_connection)
        file_manager = ExtractionFileManager()

        extractor = TaxonomyExtractor(
            taxonomy_repo=taxonomy_repo,
            chunk_repo=chunk_repo,
            file_manager=file_manager,
            grok_api_key=grok_key,
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            ollama_available=False
        )

        # Configure extractor for the selected classifier
        if initial_classifier == 'grok':
            if not extractor.grok_api_key:
                raise ValueError("Grok API key not configured")
            extractor.default_config.classifiers = ['grok']
        elif initial_classifier == 'gemini':
            if not extractor.gemini_api_key:
                raise ValueError("Gemini API key not configured")
            extractor.default_config.classifiers = ['gemini']
        elif initial_classifier == 'openai':
            if not extractor.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            extractor.default_config.classifiers = ['openai']
        elif initial_classifier == 'ollama':
            # You might want to check for Ollama availability here
            extractor.default_config.classifiers = ['ollama']
        else:
            raise ValueError(f"Unsupported classifier: {initial_classifier}")

        # Create pipeline config
        from ...features.extraction.pipeline_config import PipelineConfig
        classifiers = [initial_classifier]
        if enable_validation_stage and validation_classifier:
            classifiers.append(validation_classifier)

        pipeline_config = PipelineConfig(
            classifiers=classifiers,
            primary_classifier=initial_classifier,
            enable_coarse_stage=True,
            enable_validation_stage=enable_validation_stage,
            confidence_threshold=job_parameters.get('confidence_threshold', 0.7),
            api_keys=extractor.default_config.api_keys
        )

        progress_context.update_progress(message="Loading taxonomy data")

        # Parse taxonomy data from database
        try:
            # Parse the taxonomy JSON data
            taxonomy_dict = json.loads(taxonomy_data_json)
            if isinstance(taxonomy_dict, dict) and "taxonomy" in taxonomy_dict:
                taxonomy_raw_data = taxonomy_dict["taxonomy"]
            else:
                taxonomy_raw_data = taxonomy_dict

            # Create taxonomy loader and parse the data
            from ...features.taxonomy.loader import TaxonomyLoader
            from ...storage.src.project.database_repositories import ProjectRepository
            taxonomy_loader = TaxonomyLoader(taxonomy_repo, ProjectRepository(self.db_connection))
            taxonomy_data = taxonomy_loader._parse_taxonomy(taxonomy_raw_data)
        except Exception as e:
            raise ValueError(f"Failed to parse taxonomy data: {e}")

        progress_context.update_progress(message="Starting extraction process")

        # Perform extraction
        try:
            results, summary = extractor.extract_content(
                project_id=project_id,
                taxonomy_project=project_name,
                taxonomy_name=None,
                selected_categories=selected_categories,
                confidence_threshold=job_parameters.get('confidence_threshold', 0.5),
                max_chunks=job_parameters.get('max_chunks'),
                pipeline_config=pipeline_config,
                taxonomy_data=taxonomy_data,
                initial_classifier=initial_classifier,
                extraction_type=extraction_type,
                enable_validation_stage=enable_validation_stage,
                validation_classifier=validation_classifier,
                extraction_mode=extraction_mode
            )
        except Exception as e:
            raise ValueError(f"Extraction failed: {e}")

        progress_context.update_progress(message="Extraction completed, storing results")

        # Convert results for storage
        formatted_results = [
            {
                'chunk_id': r.chunk_id,
                'classifications': r.extracted_data.get('classifications', {}),
                'confidence_score': r.confidence,
                'categories_matched': r.categories,
                'metadata': r.extracted_data
            } for r in results
        ]

        # Store results using hybrid storage
        filesystem_manager = FilesystemStorageManager(file_manager)
        result_repo = ExtractionResultRepository(self.db_connection)
        hybrid_manager = HybridStorageManager(
            filesystem_manager=filesystem_manager,
            database_repo=result_repo,
            enable_filesystem=True
        )

        storage_result = hybrid_manager.store_results(
            job_id=job_id,
            project_id=project_id,
            taxonomy_id=taxonomy_id,
            results=formatted_results,
            storage_preference="hybrid"
        )

        progress_context.update_progress_percent(100, "Extraction job completed")

        return {
            "status": "completed",
            "operation": "extraction",
            "project_id": project_id,
            "taxonomy_id": taxonomy_id,
            "total_results": len(formatted_results),
            "extraction_summary": {
                "processed_chunks": summary.processed_chunks,
                "extraction_time": summary.extraction_time
            },
            "storage_result": storage_result,
            **parameters
        }

    def _priority_to_rq(self, priority: JobPriority) -> int:
        """Convert JobPriority to RQ priority."""
        priority_map = {
            JobPriority.LOW: 0,
            JobPriority.NORMAL: 1,
            JobPriority.HIGH: 2,
            JobPriority.URGENT: 3
        }
        return priority_map[priority]

    def get_job_status(self, job_id: str) -> Optional[EnhancedExtractionJob]:
        """Get job status by ID from Redis."""
        job_data = self.redis.get(f"compileo:job:{job_id}")
        if job_data:
            return EnhancedExtractionJob.from_dict(json.loads(job_data))
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job by ID."""
        from rq.job import Job

        # Get job from Redis
        job = self.get_job_status(job_id)
        if not job:
            return False

        if job.status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SCHEDULED]:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()

            # Update job in Redis
            self.redis.set(f"compileo:job:{job_id}", json.dumps(job.to_dict()))

            # Schedule cleanup of cancelled job after retention period
            cleanup_delay = 24 * 60 * 60  # 24 hours retention for cancelled jobs
            self.redis.expire(f"compileo:job:{job_id}", cleanup_delay)

            # Invalidate stats cache
            self._invalidate_stats_cache()

            # Remove from scheduler if scheduled
            self.scheduler.unschedule_job(job_id)

            # Try to cancel and delete RQ job
            try:
                rq_job = Job.fetch(job_id, connection=self.redis)
                if rq_job:
                    rq_job.cancel()
                    rq_job.delete()  # Also delete the job to prevent re-processing
                    logger.debug(f"Cancelled and deleted RQ job {job_id}")
            except Exception as cancel_error:
                logger.warning(f"Failed to cancel RQ job {job_id}: {cancel_error}")

            logger.info(f"Job {job_id} cancelled")
            return True
        return False

    def _get_jobs_from_redis(self, status_filter: Optional[JobStatus] = None, limit: Optional[int] = None) -> List[EnhancedExtractionJob]:
        """Get jobs from Redis with optional status filtering and limiting."""
        jobs = []
        cursor = 0

        # Use SCAN to efficiently iterate through job keys
        while True:
            cursor, keys = self.redis.scan(cursor, match="compileo:job:*", count=100)
            for key in keys:
                try:
                    # Skip processing lock keys (they contain worker info, not job data)
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                    if key_str.endswith(':processing'):
                        continue

                    job_data = self.redis.get(key)
                    if job_data:
                        try:
                            job = EnhancedExtractionJob.from_dict(json.loads(job_data))
                            if status_filter is None or job.status == status_filter:
                                jobs.append(job)
                                if limit and len(jobs) >= limit:
                                    return jobs
                        except Exception:
                            # Skip invalid job data without spamming logs
                            continue
                except Exception as e:
                    logger.debug(f"Failed to deserialize job from Redis key {key}: {e}")
                    continue

            if cursor == 0:
                break

        return jobs

    def get_pending_jobs(self, limit: int = 50) -> List[EnhancedExtractionJob]:
        """Get pending jobs from Redis."""
        pending = self._get_jobs_from_redis(JobStatus.PENDING, limit)
        return sorted(pending, key=lambda j: j.created_at, reverse=True)

    def get_running_jobs(self, limit: int = 50) -> List[EnhancedExtractionJob]:
        """Get running jobs from Redis."""
        running = self._get_jobs_from_redis(JobStatus.RUNNING, limit)
        return sorted(running, key=lambda j: j.started_at or j.created_at, reverse=True)

    def get_jobs_by_status(self, status: JobStatus, limit: int = 50) -> List[EnhancedExtractionJob]:
        """Get jobs by status from Redis."""
        return self._get_jobs_from_redis(status, limit)




    def _invalidate_stats_cache(self) -> None:
        """Invalidate the statistics cache."""
        self._stats_cache = None
        self._stats_cache_time = None

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics from Redis with caching."""
        # Check cache first
        now = datetime.utcnow()
        if (self._stats_cache and self._stats_cache_time and
            (now - self._stats_cache_time).total_seconds() < self._stats_cache_ttl):
            # Update dynamic fields that change frequently
            self._stats_cache.update({
                'active_workers': self.resource_manager.active_jobs,
                'cpu_usage_percent': self.resource_manager.current_cpu_percent,
                'memory_usage_mb': self.resource_manager.current_memory_mb
            })
            return self._stats_cache

        # Count jobs by status from Redis
        status_counts = {
            JobStatus.PENDING: 0,
            JobStatus.RUNNING: 0,
            JobStatus.COMPLETED: 0,
            JobStatus.FAILED: 0,
            JobStatus.CANCELLED: 0,
            JobStatus.RETRYING: 0,
            JobStatus.SCHEDULED: 0
        }

        cursor = 0
        total_jobs = 0

        # Use SCAN to efficiently count jobs by status
        while True:
            cursor, keys = self.redis.scan(cursor, match="compileo:job:*", count=100)
            for key in keys:
                try:
                    # Skip processing lock keys (they contain worker info, not job data)
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                    if key_str.endswith(':processing'):
                        continue

                    job_data = self.redis.get(key)
                    if job_data:
                        job = EnhancedExtractionJob.from_dict(json.loads(job_data))
                        status_counts[job.status] = status_counts.get(job.status, 0) + 1
                        total_jobs += 1
                except Exception as e:
                    logger.warning(f"Failed to deserialize job from Redis key {key}: {e}")
                    continue

            if cursor == 0:
                break

        stats = {
            'pending_jobs': status_counts[JobStatus.PENDING],
            'running_jobs': status_counts[JobStatus.RUNNING],
            'scheduled_jobs': len(self.scheduler.scheduled_jobs),
            'completed_jobs': status_counts[JobStatus.COMPLETED],
            'failed_jobs': status_counts[JobStatus.FAILED],
            'cancelled_jobs': status_counts[JobStatus.CANCELLED],
            'retrying_jobs': status_counts[JobStatus.RETRYING],
            'total_jobs': total_jobs,
            'queue_type': 'redis',
            'cache_size': len(self.result_cache.cache),
            'cpu_usage_percent': self.resource_manager.current_cpu_percent,
            'memory_usage_mb': self.resource_manager.current_memory_mb
        }

        # Cache the results
        self._stats_cache = stats
        self._stats_cache_time = now

        return stats




class EnhancedJobQueueManager:
    """Enhanced central manager for job queues with advanced features."""

    def __init__(
        self,
        redis_url: str = 'redis://localhost:6379/0',
        db_connection=None,
        global_max_jobs: int = 10,
        enable_background_monitoring: bool = True
    ):
        from redis import Redis
        from rq import Queue

        self.queue = EnhancedRedisJobQueue(
            redis_url,
            db_connection=db_connection,
            global_max_jobs=global_max_jobs,
            enable_background_monitoring=enable_background_monitoring
        )

    def submit_job(
        self,
        job_type: Union[str, JobType],
        parameters: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        schedule: Optional[JobSchedule] = None,
        dependencies: Optional[List[JobDependency]] = None,
        resource_limits: Optional[ResourceLimits] = None,
        tags: Optional[Set[str]] = None,
        cache_key: Optional[str] = None,
        parent_job_id: Optional[str] = None,
        user_id: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> str:
        """Submit an enhanced job with full feature support."""
        from datetime import datetime
        # DEBUG: [DEBUG_20251003_PDF_PARSE] - EnhancedJobQueueManager submit_job called
        import json
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "enhanced_job_queue_manager",
            "message": "EnhancedJobQueueManager submit_job called - STARTING JOB CREATION",
            "context": {
                "job_type": str(job_type),
                "user_id": user_id or "unknown",
                "parameters_keys": list(parameters.keys()),
                "operation": parameters.get("operation", "unknown")
            }
        }
        logger.debug(json.dumps(debug_context))
        logger.debug(f"submit_job called for operation: {parameters.get('operation', 'unknown')}")

        if not job_id:
            job_id = str(uuid.uuid4())
            
        job = EnhancedExtractionJob(
            job_id=job_id,
            job_type=job_type,
            parameters=parameters,
            priority=priority,
            schedule=schedule,
            dependencies=dependencies,
            resource_limits=resource_limits,
            user_id=user_id
        )

        if tags:
            job.tags = tags
        job.cache_key = cache_key
        job.parent_job_id = parent_job_id

        # DEBUG: [DEBUG_20251003_PDF_PARSE] - Job created, calling queue.enqueue_job
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "enhanced_job_queue_manager",
            "message": "Job created, calling queue.enqueue_job",
            "context": {
                "job_id": job_id,
                "job_type": str(job_type),
                "user_id": user_id or "unknown"
            }
        }
        logger.debug(json.dumps(debug_context))

        result = self.queue.enqueue_job(job)

        # DEBUG: [DEBUG_20251003_PDF_PARSE] - Queue enqueue_job returned
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "enhanced_job_queue_manager",
            "message": "Queue enqueue_job returned",
            "context": {
                "job_id": job_id,
                "result": result,
                "queue_type": type(self.queue).__name__
            }
        }
        logger.debug(json.dumps(debug_context))

        return result

    def get_job(self, job_id: str) -> Optional[EnhancedExtractionJob]:
        """Get job by ID."""
        return self.queue.get_job_status(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        return self.queue.cancel_job(job_id)

    def restart_job(self, job_id: str) -> bool:
        """Restart a failed or cancelled job."""
        job = self.get_job(job_id)
        if not job or job.status not in [JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        # Reset job state
        job.status = JobStatus.PENDING
        job.started_at = None
        job.completed_at = None
        job.error = None
        job.progress = 0.0
        job.retry_count = 0

        # Re-enqueue
        return self.queue.enqueue_job(job) == job_id

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        return self.queue.get_queue_stats()

    def get_jobs_by_tag(self, tag: str, limit: int = 50) -> List[EnhancedExtractionJob]:
        """Get jobs by tag."""
        jobs = []
        for job in self.queue._job_storage.values() if hasattr(self.queue, '_job_storage') else []:
            if tag in job.tags:
                jobs.append(job)
                if len(jobs) >= limit:
                    break
        return jobs

    def get_jobs_by_status(self, status: JobStatus, limit: int = 50) -> List[EnhancedExtractionJob]:
        """Get jobs by status."""
        return self.queue.get_jobs_by_status(status, limit)


def initialize_job_queue_manager(
    redis_url: str = 'redis://localhost:6379/0',
    db_connection=None,
    global_max_jobs: int = 10,
    auto_start_worker: bool = False,
    enable_background_monitoring: bool = True
) -> EnhancedJobQueueManager:
    """Initialize the global job queue manager with configuration settings."""
    logger.info("Initializing job queue manager...")
    # Create the manager instance
    manager = EnhancedJobQueueManager(
        redis_url=redis_url,
        db_connection=db_connection,
        global_max_jobs=global_max_jobs,
        enable_background_monitoring=enable_background_monitoring
    )

    # Store it in the module's global namespace
    import sys
    current_module = sys.modules[__name__]
    current_module.enhanced_job_queue_manager = manager

    logger.info("Job queue manager initialized.")
    return manager


# Global enhanced job queue manager
# Note: This will be initialized with config settings when the API starts
enhanced_job_queue_manager = None



# Convenience functions for backward compatibility and ease of use
def submit_extraction_job(
    parameters: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
    user_id: Optional[str] = None,
    **kwargs
) -> str:
    """Submit a document processing job (parsing, chunking, etc.)."""
    return enhanced_job_queue_manager.submit_job(
        JobType.DOCUMENT_PROCESSING,
        parameters,
        priority=priority,
        user_id=user_id,
        **kwargs
    )


def submit_batch_extraction_job(
    parameters: Dict[str, Any],
    priority: JobPriority = JobPriority.HIGH,
    user_id: Optional[str] = None,
    **kwargs
) -> str:
    """Submit a batch extraction job."""
    return enhanced_job_queue_manager.submit_job(
        JobType.BATCH_EXTRACTION,
        parameters,
        priority=priority,
        user_id=user_id,
        **kwargs
    )


def submit_taxonomy_processing_job(
    parameters: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
    user_id: Optional[str] = None,
    **kwargs
) -> str:
    """Submit a taxonomy processing job."""
    return enhanced_job_queue_manager.submit_job(
        JobType.TAXONOMY_PROCESSING,
        parameters,
        priority=priority,
        user_id=user_id,
        **kwargs
    )


def submit_dataset_generation_job(
    parameters: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
    user_id: Optional[str] = None,
    **kwargs
) -> str:
    """Submit a dataset generation job."""
    return enhanced_job_queue_manager.submit_job(
        JobType.DATASET_GENERATION,
        parameters,
        priority=priority,
        user_id=user_id,
        **kwargs
    )


def submit_scheduled_job(
    job_type: JobType,
    parameters: Dict[str, Any],
    scheduled_time: datetime,
    **kwargs
) -> str:
    """Submit a job scheduled for future execution."""
    schedule = JobSchedule(scheduled_time=scheduled_time)
    return enhanced_job_queue_manager.submit_job(
        job_type,
        parameters,
        schedule=schedule,
        **kwargs
    )


def start_enhanced_worker(
    redis_url: str = 'redis://localhost:6379/0',
    queue_name: str = 'extraction_jobs',
    worker_name: Optional[str] = None,
    default_worker_ttl: int = 270000  # 75 hours for long-running jobs (72h jobs + buffer)
):
    """Start an enhanced RQ worker for processing jobs."""
    from redis import Redis
    from rq import SimpleWorker
    import time

    worker_name = worker_name or f"enhanced_worker_{os.getpid()}"
    logger.info(f"Starting enhanced worker {worker_name} with TTL {default_worker_ttl}s")

    redis_conn = Redis.from_url(redis_url)

    # CRITICAL: Clean up ANY active worker named exactly like us before starting.
    # This specifically addresses the ValueError: There exists an active worker... already
    try:
        from rq import Worker
        active_workers = Worker.all(connection=redis_conn)
        for w in active_workers:
            if w.name == worker_name:
                log_msg = {
                    "event": "worker_name_conflict_cleanup",
                    "worker_name": worker_name,
                    "message": "Found existing active worker with identical name. Cleaning up registration to prevent ValueError."
                }
                logger.warning(f"DEBUG: {json.dumps(log_msg)}")
                redis_conn.delete(f"rq:worker:{w.name}")
    except Exception as cleanup_err:
        logger.warning(f"Failed to check/cleanup existing worker name {worker_name}: {cleanup_err}")

    # Restart worker if it exits unexpectedly
    while True:
        try:
            # Listen to both main queue and intermediate queue to handle serialization issues
            # Set worker TTL to allow long-running jobs (2 hours default)
            # Use SimpleWorker for RQ 2.x to run jobs in the same process (no forking)
            worker = SimpleWorker(
                [queue_name, f"{queue_name}:intermediate"],
                connection=redis_conn,
                name=worker_name,
                default_worker_ttl=default_worker_ttl
            )

            # DEBUG: [DEBUG_20251010_WORKER_LIFECYCLE] - About to call worker.work()
            logger.info(f"Worker {worker_name} about to call worker.work()")

            # In RQ 2.x, setup_logs is removed. Centralized logging is respected automatically
            # if handlers are already configured.
            worker.work()
            # DEBUG: [DEBUG_20251010_WORKER_LIFECYCLE] - worker.work() returned (should not happen)
            logger.warning(f"Worker {worker_name} worker.work() returned unexpectedly, restarting...")
            time.sleep(1)  # Brief pause before restart

        except Exception as e:
            # DEBUG: [DEBUG_20251010_WORKER_LIFECYCLE] - worker.work() failed with exception
            logger.error(f"Worker {worker_name} worker.work() failed: {e}", exc_info=True)
            logger.info(f"Restarting worker {worker_name} in 5 seconds...")
            time.sleep(5)  # Wait before restart


# Health check and monitoring functions
def perform_health_check() -> Dict[str, Any]:
    """Perform comprehensive health check of the job queue system."""
    stats = enhanced_job_queue_manager.get_queue_stats()

    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'queue_stats': stats,
        'issues': []
    }

    # Check QueueProcessor health
    try:
        queue_processor_status = enhanced_job_queue_manager.queue.resource_manager.queue_processor.get_status()
        health_status['queue_processor'] = queue_processor_status

        if not queue_processor_status.get('running', False):
            health_status['issues'].append('QueueProcessor is not running')
        elif not queue_processor_status.get('thread_alive', False):
            health_status['issues'].append('QueueProcessor thread is not alive')
    except Exception as e:
        health_status['issues'].append(f'QueueProcessor health check failed: {e}')

    # Check for issues
    if stats.get('failed_jobs', 0) > stats.get('total_jobs', 1) * 0.1:  # >10% failure rate
        health_status['issues'].append('High job failure rate detected')

    if stats.get('cpu_usage_percent', 0) > 90:
        health_status['issues'].append('High CPU usage detected')

    if stats.get('memory_usage_mb', 0) > 900:  # >900MB
        health_status['issues'].append('High memory usage detected')

    if len(health_status['issues']) > 0:
        health_status['status'] = 'warning' if len(health_status['issues']) == 1 else 'critical'

    return health_status


def cleanup_old_jobs(days_old: int = 30) -> int:
    """Clean up old completed/failed/cancelled jobs from Redis and RQ registries."""
    if not enhanced_job_queue_manager:
        logger.warning("Job queue manager not initialized, cannot cleanup jobs")
        return 0

    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    cleaned_count = 0

    try:
        # Get all job keys from Redis
        queue = enhanced_job_queue_manager.queue
        cursor = 0
        job_keys = []

        # Use SCAN to efficiently get all job keys
        while True:
            cursor, keys = queue.redis.scan(cursor, match="compileo:job:*", count=100)
            job_keys.extend(keys)
            if cursor == 0:
                break

        logger.info(f"Found {len(job_keys)} total custom jobs in Redis")

        for key_bytes in job_keys:
            try:
                key = key_bytes.decode('utf-8')
                # Skip processing lock keys (they contain worker info, not job data)
                if key.endswith(':processing'):
                    continue

                job_data = queue.redis.get(key_bytes)

                if job_data:
                    job = EnhancedExtractionJob.from_dict(json.loads(job_data))

                    # Clean up jobs that are completed, failed, or cancelled and older than cutoff
                    if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                        job.completed_at and job.completed_at < cutoff_date):

                        # Remove from Redis
                        queue.redis.delete(key_bytes)
                        cleaned_count += 1

                        # Remove from completed jobs set if present
                        if hasattr(queue, '_completed_jobs') and job.job_id in queue._completed_jobs:
                            queue._completed_jobs.discard(job.job_id)

            except Exception as e:
                logger.warning(f"Failed to process job key {key_bytes}: {e}")
                continue

        # Clean up RQ failed jobs
        rq_failed_cleaned = queue._cleanup_rq_failed_jobs()
        cleaned_count += rq_failed_cleaned

        # Invalidate stats cache after cleanup
        if hasattr(queue, '_invalidate_stats_cache'):
            queue._invalidate_stats_cache()

        logger.info(f"Cleaned up {cleaned_count} old jobs (older than {days_old} days, including {rq_failed_cleaned} RQ failed jobs)")
        return cleaned_count

    except Exception as e:
        logger.error(f"Failed to cleanup old jobs: {e}")
        return 0


# Export key classes and functions
__all__ = [
    'JobStatus', 'JobPriority', 'JobType', 'ResourceType',
    'ResourceLimits', 'JobDependency', 'JobSchedule', 'JobMetrics',
    'EnhancedExtractionJob', 'ResourceManager', 'JobScheduler', 'ResultCache',
    'EnhancedJobQueueInterface', 'EnhancedRedisJobQueue',
    'EnhancedJobQueueManager', 'enhanced_job_queue_manager',
    'submit_extraction_job', 'submit_batch_extraction_job', 'submit_taxonomy_processing_job',
    'submit_dataset_generation_job', 'submit_scheduled_job', 'start_enhanced_worker',
    'perform_health_check', 'cleanup_old_jobs'
]