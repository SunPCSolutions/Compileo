"""
Progress Monitoring Framework for Compileo Job System.

Provides a generic ProgressContext class that enables scalable progress monitoring
for any job type.
"""

import json
import threading
from .models import EnhancedExtractionJob

class ProgressContext:
    """Generic progress monitoring context for any job type."""

    def __init__(self, job: EnhancedExtractionJob, redis_client):
        self.job = job
        self.redis = redis_client
        self.total_steps = 0
        self.current_step = 0
        self._lock = threading.Lock()

    def set_total_steps(self, steps: int):
        """Set total number of steps for this job"""
        with self._lock:
            self.total_steps = steps

    def update_progress(self, step_increment: int = 1, message: str = ""):
        """Update progress by incrementing steps"""
        with self._lock:
            self.current_step += step_increment
            progress_percent = 0.0
            if self.total_steps > 0:
                progress_percent = min(100.0, (self.current_step / self.total_steps) * 100)

            # Update job object
            self.job.update_progress(progress_percent, message)

            # Save to Redis immediately (with error handling)
            try:
                self.redis.set(f"compileo:job:{self.job.job_id}", json.dumps(self.job.to_dict()))
            except Exception as e:
                # Log error but don't fail the job execution
                # In production, this would be logged to a proper logging system
                pass

    def update_progress_percent(self, percent: float, message: str = ""):
        """Update progress directly with percentage"""
        with self._lock:
            progress_percent = max(0.0, min(100.0, percent))

            # Update job object
            self.job.update_progress(progress_percent, message)

            # Save to Redis immediately (with error handling)
            try:
                self.redis.set(f"compileo:job:{self.job.job_id}", json.dumps(self.job.to_dict()))
            except Exception as e:
                # Log error but don't fail the job execution
                # In production, this would be logged to a proper logging system
                pass

    def get_current_progress(self) -> float:
        """Get current progress percentage"""
        with self._lock:
            return self.job.progress