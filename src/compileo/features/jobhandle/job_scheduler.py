"""
Job Scheduler: Handles job scheduling and time-based execution.
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


class JobScheduler:
    """Handles job scheduling and time-based execution."""

    def __init__(self):
        self.scheduled_jobs: Dict[str, EnhancedExtractionJob] = {}
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False

    def schedule_job(self, job: EnhancedExtractionJob) -> None:
        """Schedule a job for future execution."""
        if job.schedule.scheduled_time:
            self.scheduled_jobs[job.job_id] = job
            job.status = JobStatus.SCHEDULED
            logger.info(f"Scheduled job {job.job_id} for {job.schedule.scheduled_time}")

    def unschedule_job(self, job_id: str) -> bool:
        """Remove job from schedule."""
        if job_id in self.scheduled_jobs:
            del self.scheduled_jobs[job_id]
            return True
        return False

    def get_due_jobs(self) -> List[EnhancedExtractionJob]:
        """Get jobs that are due for execution."""
        now = datetime.utcnow()
        due_jobs = []

        for job in self.scheduled_jobs.values():
            if job.schedule.scheduled_time and job.schedule.scheduled_time <= now:
                due_jobs.append(job)

        # Remove scheduled jobs that are now due
        for job in due_jobs:
            del self.scheduled_jobs[job.job_id]

        return due_jobs

    def start_scheduler(self) -> None:
        """Start the scheduler thread."""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Job scheduler started")

    def stop_scheduler(self) -> None:
        """Stop the scheduler thread."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("Job scheduler stopped")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                due_jobs = self.get_due_jobs()
                for job in due_jobs:
                    # Queue job for execution
                    job.status = JobStatus.PENDING
                    logger.info(f"Job {job.job_id} is now due for execution")

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(30)  # Wait longer on error