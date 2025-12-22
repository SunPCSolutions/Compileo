"""
Queue Processor: Handles processing of queued jobs when resources become available.
"""

from typing import Dict, Any, Optional, List, Callable, Set, Tuple, Union
from datetime import datetime, timedelta
import uuid
import json
import time
import threading
from ...core.logging import get_logger
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

logger = get_logger(__name__)


class QueueProcessor:
    """Processes queued jobs when resources become available."""

    def __init__(self, resource_manager: 'ResourceManager'):
        self.resource_manager = resource_manager
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        self._job_queues: List['EnhancedJobQueueInterface'] = []

    def register_queue(self, queue: 'EnhancedJobQueueInterface') -> None:
        """Register a job queue to process."""
        self._job_queues.append(queue)

    def start_processing(self) -> None:
        """Start the queue processing thread."""
        if self._running:
            logger.warning("QueueProcessor: Already running, ignoring start request")
            return

        logger.info("QueueProcessor: Starting queue processing thread")
        self._running = True
        self._processor_thread = threading.Thread(
            target=self._process_queues,
            daemon=True,
            name="QueueProcessor"
        )
        self._processor_thread.start()
        logger.info("QueueProcessor: Queue processing thread started successfully")

    def stop_processing(self) -> None:
        """Stop the queue processing thread."""
        logger.info("QueueProcessor: Stopping queue processing")
        self._running = False
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5)
            if self._processor_thread.is_alive():
                logger.warning("QueueProcessor: Thread did not stop gracefully within 5 seconds")
        logger.info("QueueProcessor: Queue processing stopped")

    def is_healthy(self) -> bool:
        """Check if the queue processor is healthy."""
        return (self._running and
                self._processor_thread is not None and
                self._processor_thread.is_alive())

    def get_status(self) -> dict:
        """Get status information about the queue processor."""
        return {
            'running': self._running,
            'thread_alive': self._processor_thread.is_alive() if self._processor_thread else False,
            'thread_name': self._processor_thread.name if self._processor_thread else None,
            'registered_queues': len(self._job_queues)
        }

    def _process_queues(self) -> None:
        """Main queue processing loop."""
        logger.info("QueueProcessor: Starting main processing loop")
        loop_count = 0

        while self._running:
            try:
                loop_count += 1
                total_pending = 0
                queues_processed = 0

                for queue in self._job_queues:
                    try:
                        queues_processed += 1
                        pending_count = self._process_queue(queue)
                        total_pending += pending_count
                    except Exception as queue_error:
                        logger.error(f"QueueProcessor: Error processing queue {queues_processed}: {queue_error}", exc_info=True)

                # Log activity periodically (every 10 loops = 50 seconds)
                # if loop_count % 10 == 0:
                #     logger.debug(f"QueueProcessor: Status - processed {queues_processed} queues, found {total_pending} pending jobs in last cycle")

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"QueueProcessor: Critical error in main loop: {e}", exc_info=True)
                time.sleep(10)  # Wait longer on critical errors

    def _process_queue(self, queue: 'EnhancedJobQueueInterface') -> int:
        """Process pending jobs in a queue. Returns number of jobs processed."""
        try:
            pending_jobs = queue.get_pending_jobs(limit=10)  # Process up to 10 at a time
            jobs_processed = 0

            if pending_jobs:
                logger.debug(f"QueueProcessor: Found {len(pending_jobs)} pending jobs in queue")

            for job in pending_jobs:
                try:
                    # Check if job can now be started
                    if self.resource_manager.check_resource_limits(job.resource_limits, job.user_id):
                        # Try to start the job
                        if self._start_queued_job(queue, job):
                            logger.info(f"QueueProcessor: Started queued job {job.job_id}")
                            jobs_processed += 1
                        else:
                            logger.info(f"QueueProcessor: Failed to start queued job {job.job_id} (check debug logs)")
                    else:
                        logger.debug(f"QueueProcessor: Job {job.job_id} waiting for resources")
                except Exception as job_error:
                    logger.error(f"QueueProcessor: Error processing individual job {job.job_id}: {job_error}")

            return jobs_processed

        except Exception as e:
            logger.error(f"QueueProcessor: Error processing queue: {e}", exc_info=True)
            return 0

    def _start_queued_job(self, queue: 'EnhancedJobQueueInterface', job: EnhancedExtractionJob) -> bool:
        """Start a queued job through Redis/RQ only."""
        try:
            logger.info(f"QueueProcessor: Starting queued job {job.job_id} (status: {job.status})")

            # Use a scheduling lock to prevent race conditions between multiple QueueProcessors
            # (e.g. if multiple API instances are running or if worker accidentally spawns one)
            scheduling_lock_key = f"compileo:job:{job.job_id}:scheduling_lock"
            if hasattr(queue, 'redis'):
                # Try to acquire lock for 10 seconds
                if not queue.redis.set(scheduling_lock_key, "locked", ex=10, nx=True):
                    logger.debug(f"QueueProcessor: Could not acquire scheduling lock for job {job.job_id}, another processor handling it")
                    return False

            try:
                # Re-fetch job data to ensure it is still PENDING (atomic check pattern)
                # This prevents double-processing if another processor just picked it up
                if hasattr(queue, 'get_job_status'):
                    current_job = queue.get_job_status(job.job_id)
                    if current_job and current_job.status != JobStatus.PENDING:
                        logger.debug(f"QueueProcessor: Job {job.job_id} is no longer PENDING (status: {current_job.status}), skipping")
                        return False

                # FIRST: Check if job is already enqueued in RQ to prevent duplicate processing
                # This check must happen BEFORE any status changes
                if hasattr(queue, 'queue'):
                    try:
                        from rq.job import Job
                        rq_job = Job.fetch(job.job_id, connection=queue.redis)
                        if rq_job:
                            logger.debug(f"QueueProcessor: Job {job.job_id} already exists in RQ, skipping to prevent duplicate processing")
                            return False
                    except Exception as e:
                        # Job doesn't exist in RQ, which is fine - we can proceed
                        logger.debug(f"QueueProcessor: Job {job.job_id} not found in RQ (expected), proceeding with queue processing")

                # Update job status to running
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()

                # Save updated status to Redis
                if hasattr(queue, 'redis'):
                    queue.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict()))
                    logger.debug(f"DEBUG_QUEUE_PROCESSOR: Saved job {job.job_id} status as RUNNING in Redis")

            finally:
                # Release scheduling lock
                if hasattr(queue, 'redis'):
                    queue.redis.delete(scheduling_lock_key)

            # All jobs now go through Redis queue exclusively
            if hasattr(queue, 'queue'):
                # This is a Redis queue - use standalone function
                rq_priority = queue._priority_to_rq(job.priority)
                logger.debug(f"DEBUG_QUEUE_PROCESSOR: Enqueueing job {job.job_id} in RQ with priority {rq_priority}")
                queue.queue.enqueue(
                    'src.compileo.features.jobhandle.worker_wrapper.process_job',
                    job.job_id,
                    priority=rq_priority,
                    timeout=job.schedule.timeout_seconds
                )
                logger.debug(f"DEBUG_QUEUE_PROCESSOR: Successfully enqueued job {job.job_id} in RQ")
                return True
            else:
                logger.error(f"Job {job.job_id} requires Redis queue, but queue type not supported")
                return False

        except Exception as e:
            logger.error(f"Failed to start queued job {job.job_id}: {e}")
            job.status = JobStatus.PENDING  # Reset status
            # Save reset status to Redis
            if hasattr(queue, 'redis'):
                queue.redis.set(f"compileo:job:{job.job_id}", json.dumps(job.to_dict()))
            return False