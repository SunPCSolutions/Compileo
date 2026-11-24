"""
Unified Job Service for Compileo GUI.

Provides a consistent interface for job submission and synchronous monitoring
throughout the Compileo application. This service implements the wizard's
successful pattern of inline progress display and synchronous polling.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import streamlit as st

from ..services.api_client import api_client
from ...jobhandle.models import JobType, JobPriority

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class JobResult:
    """
    Result of a job submission and monitoring operation.
    
    Attributes:
        job_id: Unique identifier for the job
        status: Final status of the job
        result: Optional result data from completed job
        error_message: Error message if job failed
        start_time: When the job was submitted
        end_time: When the job completed/failed/timed out
    """
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED.value
    
    @property
    def failed(self) -> bool:
        """Check if job failed."""
        return self.status == JobStatus.FAILED.value
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def __str__(self) -> str:
        if self.success:
            return f"Job {self.job_id}: Success ({self.duration_seconds:.1f}s)"
        elif self.error_message:
            return f"Job {self.job_id}: Failed - {self.error_message}"
        else:
            return f"Job {self.job_id}: {self.status}"


class UnifiedJobService:
    """
    Unified service for job submission and synchronous monitoring.
    
    This service provides a consistent interface for submitting jobs and
    monitoring their progress with inline UI updates. It implements the
    wizard's successful pattern throughout Compileo.
    
    Example usage:
        service = UnifiedJobService()
        result = service.submit_and_monitor(
            job_type=JobType.DOCUMENT_PROCESSING,
            parameters={"project_id": 123, "document_ids": [1, 2, 3]},
            title="Processing Documents",
            timeout=1800
        )
        
        if result.success:
            st.success("Documents processed!")
        else:
            st.error(f"Failed: {result.error_message}")
    """
    
    def __init__(self):
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self._default_timeout = 3600  # 1 hour
        self._default_poll_interval = 3  # 3 seconds
    
    def submit_and_monitor(
        self,
        job_type: JobType,
        parameters: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        user_id: Optional[str] = None,
        title: str = "Processing",
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        timeout: Optional[int] = None,
        poll_interval: Optional[int] = None,
        show_inline_progress: bool = True,
        show_job_id: bool = True
    ) -> JobResult:
        """
        Submit a job and monitor until completion.
        
        This is the primary method for job submission that should be used
        throughout the Compileo GUI for consistent user experience.
        
        Args:
            job_type: Type of job to submit (e.g., JobType.DOCUMENT_PROCESSING)
            parameters: Job-specific parameters dictionary
            priority: Job execution priority (default: NORMAL)
            user_id: Optional user ID for job attribution
            title: Title displayed in progress UI (default: "Processing")
            on_complete: Optional callback(result: Dict) on success
            on_error: Optional callback(error_message: str) on failure
            timeout: Maximum seconds to wait (default: 3600)
            poll_interval: Seconds between status checks (default: 3)
            show_inline_progress: Whether to show Streamlit progress UI (default: True)
            show_job_id: Whether to display job ID in UI (default: True)
            
        Returns:
            JobResult with status, result, and timing information
        """
        start_time = datetime.now()
        timeout = timeout or self._default_timeout
        poll_interval = poll_interval or self._default_poll_interval
        
        # Submit the job
        job_id = self._submit_job(
            job_type=job_type,
            parameters=parameters,
            priority=priority,
            user_id=user_id,
            start_time=start_time
        )
        
        if not job_id:
            error_msg = "Failed to submit job - no job ID returned"
            if on_error:
                on_error(error_msg)
            return JobResult(
                job_id="",
                status=JobStatus.FAILED.value,
                error_message=error_msg,
                start_time=start_time
            )
        
        # Track the job
        self._active_jobs[job_id] = {
            "start_time": start_time,
            "parameters": parameters,
            "job_type": job_type,
            "title": title
        }
        
        # Set up UI components if requested
        status_text = None
        job_id_text = None
        
        if show_inline_progress:
            if show_job_id:
                job_id_text = st.empty()
                job_id_text.info(f"🚀 {title} started (Job ID: `{job_id}`)")
            else:
                st.info(f"🚀 {title} started")
            status_text = st.empty()
        
        # Monitor the job
        return self._monitor_job(
            job_id=job_id,
            start_time=start_time,
            status_text=status_text,
            job_id_text=job_id_text,
            on_complete=on_complete,
            on_error=on_error,
            timeout=timeout,
            poll_interval=poll_interval
        )
    
    def _submit_job(
        self,
        job_type: JobType,
        parameters: Dict[str, Any],
        priority: JobPriority,
        user_id: Optional[str],
        start_time: datetime
    ) -> Optional[str]:
        """
        Submit a job to the job queue.
        
        Returns:
            Job ID if successful, None otherwise
        """
        try:
            from src.compileo.features.gui.services.async_job_submission_service import async_job_service
            
            job_id = async_job_service.submit_job(
                job_type=job_type,
                parameters=parameters,
                priority=priority,
                user_id=user_id
            )
            
            logger.info(f"Job {job_id} submitted successfully")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            return None
    
    def _monitor_job(
        self,
        job_id: str,
        start_time: datetime,
        status_text=None,
        job_id_text=None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        timeout: int = 3600,
        poll_interval: int = 3
    ) -> JobResult:
        """
        Monitor job status until completion or timeout.
        
        Returns:
            JobResult with final status and data
        """
        end_time = None
        final_status = None
        result = None
        error_message = None
        
        try:
            while True:
                # Check for timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    error_message = f"Job timed out after {timeout} seconds"
                    if status_text:
                        status_text.error(f"⏰ {error_message}")
                    final_status = JobStatus.TIMEOUT.value
                    break
                
                # Get status from API
                status_response = self._get_job_status(job_id)
                
                if not status_response:
                    # API error - will retry on next iteration
                    time.sleep(poll_interval)
                    continue
                
                status = status_response.get("status", "unknown")
                current_step = status_response.get("current_step", "Processing...")
                
                # Update UI - status only, no percentage
                if status_text:
                    if status == "running":
                        status_text.info(f"🔄 {current_step}")
                    elif status == "pending":
                        status_text.warning(f"⏳ Job pending...")
                    elif status == "queued":
                        status_text.warning(f"⏳ Job queued...")
                
                # Check completion states
                if status == "completed":
                    result = status_response.get("result", {})
                    final_status = JobStatus.COMPLETED.value
                    end_time = datetime.now()
                    
                    if status_text:
                        status_text.success("✅ Job completed successfully!")
                    if job_id_text:
                        job_id_text.empty()
                    if on_complete:
                        try:
                            on_complete(result)
                        except Exception as callback_error:
                            logger.warning(f"Complete callback error: {callback_error}")
                    break
                    
                elif status == "failed":
                    error_message = status_response.get("error", "Unknown error")
                    final_status = JobStatus.FAILED.value
                    end_time = datetime.now()
                    
                    if status_text:
                        status_text.error(f"❌ Job failed: {error_message}")
                    if job_id_text:
                        job_id_text.empty()
                    if on_error:
                        try:
                            on_error(error_message)
                        except Exception as callback_error:
                            logger.warning(f"Error callback error: {callback_error}")
                    break
                    
                elif status == "cancelled":
                    final_status = JobStatus.CANCELLED.value
                    end_time = datetime.now()
                    
                    if status_text:
                        status_text.warning("🚫 Job was cancelled")
                    if job_id_text:
                        job_id_text.empty()
                    break
                
                # Wait before next poll
                time.sleep(poll_interval)
        
        except Exception as e:
            error_message = f"Monitoring error: {str(e)}"
            final_status = JobStatus.FAILED.value
            end_time = datetime.now()
            
            logger.error(f"Error monitoring job {job_id}: {e}")
            
            if status_text:
                status_text.error(f"❌ {error_message}")
            if job_id_text:
                job_id_text.empty()
            if on_error:
                try:
                    on_error(error_message)
                except Exception as callback_error:
                    logger.warning(f"Error callback error: {callback_error}")
        
        finally:
            # Clean up active jobs
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
        
        return JobResult(
            job_id=job_id,
            status=final_status or JobStatus.UNKNOWN.value,
            result=result,
            error_message=error_message,
            start_time=start_time,
            end_time=end_time
        )
    
    def _get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status from API.
        
        Returns:
            Status dictionary or None if request failed
        """
        try:
            return api_client.get(f"/api/v1/jobs/status/{job_id}")
        except Exception as e:
            logger.warning(f"Failed to get status for job {job_id}: {e}")
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: ID of job to cancel
            
        Returns:
            True if cancellation was successful
        """
        try:
            from src.compileo.features.gui.services.async_job_submission_service import async_job_service
            return async_job_service.cancel_job(job_id)
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get currently monitored jobs.
        
        Returns:
            Dictionary of job_id -> job_info
        """
        return self._active_jobs.copy()


# Global instance for convenience
unified_job_service = UnifiedJobService()


def submit_and_monitor_job(
    job_type: JobType,
    parameters: Dict[str, Any],
    title: str = "Processing",
    **kwargs
) -> JobResult:
    """
    Convenience function to submit and monitor a job using the global service.
    
    Args:
        job_type: Type of job to submit
        parameters: Job parameters
        title: Title for progress display
        **kwargs: Additional arguments passed to submit_and_monitor
        
    Returns:
        JobResult with status and data
    """
    return unified_job_service.submit_and_monitor(
        job_type=job_type,
        parameters=parameters,
        title=title,
        **kwargs
    )
