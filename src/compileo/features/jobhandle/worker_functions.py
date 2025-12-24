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
from ...core.logging import setup_logging, get_logger

from .models import (
    JobStatus, JobPriority, JobType, ResourceType,
    ResourceLimits, JobDependency, JobSchedule, JobMetrics,
    EnhancedExtractionJob
)

logger = get_logger(__name__)


def redis_job_wrapper(job_id: str, **kwargs) -> Any:
    """Standalone job wrapper function for RQ that can be pickled."""
    from datetime import datetime
    # Initialize logging for the worker process
    setup_logging()
    
    logger.debug(f"redis_job_wrapper called with job_id={job_id}")
    # DEBUG: [DEBUG_20251003_RQ_FLOW] - RQ worker picked up a job.
    debug_context = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "component": "redis_job_wrapper",
        "message": "redis_job_wrapper started",
        "context": {
            "job_id": job_id,
            "kwargs_keys": list(kwargs.keys())
        }
    }
    logger.debug(json.dumps(debug_context))

    logger.info(f"redis_job_wrapper started for job_id: {job_id}")
    
    # Initialize a new EnhancedJobQueueManager for this worker process
    # This ensures the worker has its own context and doesn't rely on a shared global
    from .enhanced_job_queue import EnhancedJobQueueManager, enhanced_job_queue_manager
    from src.compileo.core.settings import BackendSettings

    # Prefer the global manager if initialized (which has correct limits and settings from worker.py)
    if enhanced_job_queue_manager:
        manager = enhanced_job_queue_manager
        logger.debug(f"Using existing global EnhancedJobQueueManager for job {job_id}")
    else:
        # Fallback: Initialize with correct settings if global is missing
        backend_settings = BackendSettings()
        global_max_jobs = backend_settings.get_global_max_concurrent_jobs()
        # TODO: Uncomment when multi-user architecture is implemented
        # per_user_max_jobs = backend_settings.get_per_user_max_concurrent_jobs()
        per_user_max_jobs = global_max_jobs  # Single-user mode
        
        manager = EnhancedJobQueueManager(
            global_max_jobs=global_max_jobs,
            enable_background_monitoring=False  # Disable background threads for per-job instances
        )
        logger.debug(f"Initialized new EnhancedJobQueueManager (fallback) for job {job_id} with limits: Global={global_max_jobs}, User={per_user_max_jobs}")

    # Get the Redis queue from the manager
    queue = manager.queue

    # Fetch job data from Redis
    debug_context = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "component": "redis_job_wrapper",
        "message": "Fetching job data from Redis",
        "context": {"job_id": job_id}
    }
    logger.debug(json.dumps(debug_context))

    job_data = queue.redis.get(f"compileo:job:{job_id}")
    if not job_data:
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "component": "redis_job_wrapper",
            "message": "Job data not found in Redis",
            "context": {"job_id": job_id}
        }
        logger.debug(json.dumps(debug_context))
        logger.error(f"Job {job_id} data not found in Redis")
        return None

    debug_context = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "component": "redis_job_wrapper",
        "message": "Job data retrieved from Redis",
        "context": {"job_id": job_id, "data_length": len(job_data)}
    }
    logger.debug(json.dumps(debug_context))

    job = EnhancedExtractionJob.from_dict(json.loads(job_data))

    # Check job parameters after retrieval
    logger.debug(f"job_id={job_id}, parameters_keys={list(job.parameters.keys()) if job.parameters else 'None'}")
    logger.debug(f"extraction_mode={job.parameters.get('extraction_mode', 'NOT_FOUND') if job.parameters else 'NO_PARAMS'}")

    debug_context = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "component": "redis_job_wrapper",
        "message": "Job object created",
        "context": {"job_id": job_id, "job_type": str(job.job_type), "status": str(job.status)}
    }
    logger.debug(json.dumps(debug_context))

    # Check if job is already in a final state to prevent duplicate execution
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        logger.warning(f"Job {job_id} is already in final state '{job.status}'. Skipping execution to prevent duplication.")
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "WARNING",
            "component": "redis_job_wrapper",
            "message": "Job already in final state, skipping execution.",
            "context": {"job_id": job_id, "status": str(job.status)}
        }
        logger.debug(json.dumps(debug_context))
        return None

    # Use Redis atomic operation to acquire processing lock
    processing_lock_key = f"compileo:job:{job_id}:processing"
    worker_lock_value = f"worker_{os.getpid()}_{datetime.utcnow().isoformat()}"

    # Try to acquire processing lock atomically (expires in 5 minutes)
    lock_acquired = queue.redis.set(processing_lock_key, worker_lock_value, ex=300, nx=True)

    if not lock_acquired:
        # Another worker is already processing this job
        existing_lock = queue.redis.get(processing_lock_key)
        # Decode bytes for JSON serialization
        existing_lock_decoded = existing_lock.decode('utf-8') if isinstance(existing_lock, bytes) else str(existing_lock) if existing_lock else None
        logger.warning(f"Job {job_id} is already being processed by another worker (lock: {existing_lock_decoded}). Skipping execution.")
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "WARNING",
            "component": "redis_job_wrapper",
            "message": "Job already being processed by another worker, skipping execution.",
            "context": {
                "job_id": str(job_id),
                "existing_lock": existing_lock_decoded
            }
        }
        logger.debug(json.dumps(debug_context))
        return None

    # Re-initialize the db_connection for the worker process
    if not queue.db_connection:
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "redis_job_wrapper",
            "message": "Initializing database connection",
            "context": {"job_id": job_id}
        }
        logger.debug(json.dumps(debug_context))

        from src.compileo.storage.src.database import get_db_connection
        queue.db_connection = get_db_connection()
        logger.info("Database connection re-initialized in worker.")

    logger.info(f"Job {job_id} retrieved from storage. Status: {job.status}")

    # --- START: Critical Section for Status Update and Resource Allocation ---
    # Job status checks were already performed above, proceed with execution

    # Update job status to RUNNING immediately to prevent other workers from starting
    logger.info(f"Setting job {job_id} status to RUNNING (Pre-Resource Check)")
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()
    job.worker_id = f"worker_{os.getpid()}"
    # Save RUNNING status to Redis immediately
    queue.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict()))
    # Note: Don't store RQ job data here - let RQ handle its own storage

    # Update job status in database to RUNNING (sets started_at)
    try:
        from src.compileo.storage.src.project.database_repositories import ExtractionJobRepository
        if queue.db_connection:
            # Ensure connection is alive
            try:
                queue.db_connection.cursor().execute("SELECT 1")
            except Exception:
                from src.compileo.storage.src.database import get_db_connection
                queue.db_connection = get_db_connection()
                logger.info(f"Reconnected to database for job {job_id} (start)")

            job_repo = ExtractionJobRepository(queue.db_connection)
            job_repo.update_job_status(job_id, "running")
            queue.db_connection.commit()
            logger.info(f"Successfully committed job {job_id} status 'running' to database")
    except Exception as db_error:
        logger.error(f"Failed to update job {job_id} status to RUNNING in database: {db_error}")

    try:
        logger.info(f"Checking resource limits for job {job_id}")
        # Pre-execution limit check to prevent race conditions
        logger.debug(f"About to call check_resource_limits for job {job_id}")
        resource_check_result = queue.resource_manager.check_resource_limits(job.resource_limits, job.user_id)
        logger.debug(f"check_resource_limits returned {resource_check_result} for job {job_id}")
        if not resource_check_result:
            logger.warning(f"Pre-execution resource check failed for job {job_id}, resetting to PENDING")
            # Reset job status to pending so queue processor can pick it up again
            job.status = JobStatus.PENDING
            job.started_at = None
            queue.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict())) # Save PENDING status
            
            # Do NOT raise exception here, as that triggers FAILED status in the except block
            # Instead return None, which tells RQ the job is "done" (so it's removed from RQ)
            # But we saved it as PENDING in Redis, so QueueProcessor will pick it up again later
            logger.info(f"Job {job_id} requeued (pending resources)")
            return None

        # Allocate resources
        logger.debug(f"About to call allocate_resources for job {job_id}")
        logger.debug(f"queue.resource_manager = {queue.resource_manager}")
        try:
            allocate_result = queue.resource_manager.allocate_resources(job.resource_limits, job.user_id)
            logger.debug(f"allocate_resources returned {allocate_result}")
        except Exception as e:
            logger.error(f"allocate_resources failed with exception: {e}", exc_info=True)
            raise
        logger.debug(f"allocate_resources returned {allocate_result} for job {job_id}")
        if not allocate_result:
            logger.warning(f"Resource allocation failed for job {job_id}, resetting to PENDING")
            job.status = JobStatus.PENDING
            job.started_at = None
            queue.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict())) # Save PENDING status
            return None

        # Note: Job status is already RUNNING and saved to Redis. Proceed to execution.
        # --- END: Critical Section for Status Update and Resource Allocation ---

        # Execute job
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "redis_job_wrapper",
            "message": "Starting job execution",
            "context": {"job_id": job_id, "job_type": str(job.job_type)}
        }
        logger.debug(json.dumps(debug_context))

        logger.debug(f"About to check if queue has _execute_job method")
        logger.debug(f"queue type = {type(queue)}")
        logger.debug(f"queue has _execute_job = {hasattr(queue, '_execute_job')}")

        logger.info(f"Executing job {job_id}...")
        logger.debug(f"About to call queue._execute_job for job {job_id}")
        logger.debug(f"job_type={job.job_type}, parameters_keys={list(job.parameters.keys())}")
        try:
            result = queue._execute_job(job.job_type, job.parameters, job)
            logger.debug(f"queue._execute_job returned for job {job_id}")
        except Exception as execute_error:
            logger.error(f"queue._execute_job failed for job {job_id}: {execute_error}", exc_info=True)
            raise

        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "redis_job_wrapper",
            "message": "Job execution finished",
            "context": {"job_id": job_id, "result_type": str(type(result))}
        }
        logger.debug(json.dumps(debug_context))

        logger.info(f"Job {job_id} execution finished.")

        # Update job status on success in Redis
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.result = result
        queue.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict()))

        # Update job status in database with retry and explicit commit
        try:
            from ...storage.src.project.database_repositories import ExtractionJobRepository
            if queue.db_connection:
                # Ensure connection is alive
                try:
                    queue.db_connection.cursor().execute("SELECT 1")
                except Exception:
                    from src.compileo.storage.src.database import get_db_connection
                    queue.db_connection = get_db_connection()
                    logger.info(f"Reconnected to database for job {job_id}")

                job_repo = ExtractionJobRepository(queue.db_connection)
                job_repo.update_job_status(job_id, "completed")
                # Ensure commit happens
                queue.db_connection.commit()
                
                logger.info(f"Successfully committed job {job_id} status 'completed' to database")
                logger.info(f"Job {job_id} completed successfully - status updated in database and Redis")
            else:
                logger.error(f"No database connection available for job {job_id}")
        except Exception as db_error:
            logger.error(f"Failed to update job {job_id} status in database: {db_error}", exc_info=True)
            logger.error(f"Job {job_id} completed in Redis but database update failed - GUI may not show completion")

        # Release processing lock
        queue.redis.delete(processing_lock_key)

        # Cache result if cache key provided
        if job.cache_key:
            queue.result_cache.set(job.cache_key, result)

        # Mark as completed for dependency tracking
        if hasattr(queue, '_completed_jobs'):
            queue._completed_jobs.add(job_id)

        # Update metrics
        if job.started_at:
            job.metrics.execution_time_seconds = (job.completed_at - job.started_at).total_seconds()

        # Clean up the RQ job to prevent re-processing
        try:
            from rq.job import Job
            rq_job = Job.fetch(job_id, connection=queue.redis)
            if rq_job:
                rq_job.delete()
                logger.debug(f"Deleted RQ job {job_id} after successful completion")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup RQ job {job_id}: {cleanup_error}")

        # Schedule cleanup of completed job after retention period
        cleanup_delay = 24 * 60 * 60  # 24 hours retention for completed jobs
        queue.redis.expire(f"compileo:job:{job.job_id}", cleanup_delay)

        logger.info(f"Job {job_id} completed successfully")
        return result

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")

        # Release processing lock on failure
        try:
            queue.redis.delete(processing_lock_key)
        except Exception as lock_error:
            logger.warning(f"Failed to release processing lock for job {job_id}: {lock_error}")

        # Handle retries
        if job.should_retry():
            job.status = JobStatus.RETRYING
            job.retry_count += 1
            job.metrics.retry_count = job.retry_count

            # Reschedule with exponential backoff
            delay = job.schedule.retry_delay_seconds * (2 ** (job.retry_count - 1))
            retry_time = datetime.utcnow() + timedelta(seconds=delay)
            job.schedule.scheduled_time = retry_time
            queue.scheduler.schedule_job(job)
            # Save retrying job status to Redis
            queue.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict()))
            logger.info(f"Job {job_id} scheduled for retry {job.retry_count} at {retry_time}")
        else:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error = str(e)
            # Save failed job status to Redis
            logger.debug(f"Saving failed job {job_id} with status {job.status}")
            queue.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict()))
            logger.debug(f"Saved failed job {job_id}")

            # Clean up the RQ job to prevent re-processing of failed jobs
            try:
                from rq.job import Job
                rq_job = Job.fetch(job_id, connection=queue.redis)
                if rq_job:
                    rq_job.delete()
                    logger.debug(f"Deleted RQ job {job_id} after failure")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup RQ job {job_id} after failure: {cleanup_error}")

            # Schedule cleanup of failed job after retention period (longer than completed jobs for debugging)
            cleanup_delay = 7 * 24 * 60 * 60  # 7 days retention for failed jobs
            queue.redis.expire(f"compileo:job:{job.job_id}", cleanup_delay)

        raise

    finally:
        # Release resources
        queue.resource_manager.release_resources(job.user_id)
