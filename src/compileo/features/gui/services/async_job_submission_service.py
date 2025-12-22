"""
Async Job Submission Service for Compileo GUI.

Provides non-blocking job submission and status monitoring for GUI operations,
integrating with the enhanced job queue manager.
"""

import time
from typing import Dict, Any, Optional, Union, List
from ....core.logging import get_logger
from datetime import datetime, timedelta
import streamlit as st

from ...jobhandle.enhanced_job_queue import initialize_job_queue_manager, EnhancedJobQueueManager
from ...jobhandle.models import JobType, JobStatus, JobPriority
from ..state.session_state import session_state
from ..services.api_client import APIError
from ...extraction.error_logging import gui_logger
from ....api.core.config import settings

logger = get_logger(__name__)


class JobSubmissionError(Exception):
    """Exception raised for job submission errors."""

    def __init__(self, message: str, job_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.job_type = job_type
        self.details = details or {}


class AsyncJobSubmissionService:
    """
    Service for asynchronous job submission and monitoring in the GUI.

    This service provides non-blocking job submission that integrates with the
    enhanced job queue manager and provides real-time status feedback to users.
    """

    def __init__(self):
        """Initialize the job submission service."""
        self._job_queue_manager: Optional[EnhancedJobQueueManager] = None
        self._status_cache_timeout = 30  # seconds
        self._max_retry_attempts = 3

    @property
    def job_queue_manager(self) -> EnhancedJobQueueManager:
        """Get or initialize the job queue manager."""
        if self._job_queue_manager is None:
            try:
                self._job_queue_manager = initialize_job_queue_manager(
                    redis_url=settings.redis_url,
                    auto_start_worker=False
                )  # Don't start worker from service
                logger.info("Job queue manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize job queue manager: {e}")
                raise JobSubmissionError(f"Failed to initialize job system. Please start the worker separately: {e}")
        return self._job_queue_manager

    def submit_job(
        self,
        job_type: Union[str, JobType],
        parameters: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        user_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Submit a job asynchronously without blocking the GUI.

        Args:
            job_type: Type of job to submit
            parameters: Job-specific parameters
            priority: Job execution priority
            user_id: ID of the user submitting the job
            **kwargs: Additional job configuration options

        Returns:
            str: Unique job ID for tracking

        Raises:
            JobSubmissionError: If job submission fails
        """
        from datetime import datetime
        # DEBUG: [DEBUG_20251003_PDF_PARSE] - Async job service submit_job called
        import json
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "async_job_submission_service",
            "message": "Async job service submit_job called",
            "context": {
                "job_type": str(job_type),
                "parameters_keys": list(parameters.keys()),
                "priority": str(priority)
            }
        }
        logger.debug(json.dumps(debug_context))

        logger.debug(f"[DEBUG_20251003_RQ_FLOW] - GUI service `submit_job` called for job_type: {job_type}")
        try:
            try:
                priority_value = priority.value
            except AttributeError:
                priority_value = str(priority)

            gui_logger.log_operation_start("submit_job", context={
                "job_type": job_type,
                "user_id": user_id,
                "priority": priority_value
            })

            # Get user ID from session if not provided
            if user_id is None:
                try:
                    user_id = session_state.user_id
                    # Ensure we have a valid user_id
                    if not user_id:
                        user_id = "default_user"
                except:
                    # Fallback for when session state is not initialized
                    user_id = "default_user"

            # Final safety check
            if not user_id or user_id == "":
                user_id = "default_user"

            # DEBUG: [DEBUG_20251003_PDF_PARSE] - User ID resolved
            debug_context = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "async_job_submission_service",
                "message": "User ID resolved",
                "context": {
                    "user_id": user_id
                }
            }
            logger.debug(json.dumps(debug_context))

            logger.debug(f"[DEBUG_20251003_RQ_FLOW] - Submitting job with user_id: {user_id}")

            # DEBUG: [DEBUG_20251003_PDF_PARSE] - Calling job queue manager submit_job
            debug_context = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "async_job_submission_service",
                "message": "Calling job queue manager submit_job",
                "context": {
                    "job_type": str(job_type),
                    "user_id": user_id
                }
            }
            logger.debug(json.dumps(debug_context))

            # Submit the job
            job_id = self.job_queue_manager.submit_job(
                job_type=job_type,
                parameters=parameters,
                priority=priority,
                user_id=user_id,
                **kwargs
            )

            # DEBUG: [DEBUG_20251003_PDF_PARSE] - Job queue manager returned job_id
            debug_context = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "async_job_submission_service",
                "message": "Job queue manager returned job_id",
                "context": {
                    "job_id": job_id,
                    "job_type": str(job_type),
                    "user_id": user_id
                }
            }
            logger.debug(json.dumps(debug_context))

            logger.debug(f"[DEBUG_20251003_RQ_FLOW] - Job submitted to manager. Received job_id: {job_id}")

            # Cache the job status immediately
            self._cache_job_status(job_id)

            # Add to user's active jobs in session state
            self._add_to_user_jobs(job_id, job_type, user_id)

            gui_logger.log_operation_complete("submit_job", 0, context={
                "job_id": job_id,
                "job_type": str(job_type),
                "user_id": user_id
            })

            # DEBUG: [DEBUG_20251003_PDF_PARSE] - Job submission completed successfully
            debug_context = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "async_job_submission_service",
                "message": "Job submission completed successfully",
                "context": {
                    "job_id": job_id,
                    "job_type": str(job_type),
                    "user_id": user_id
                }
            }
            logger.debug(json.dumps(debug_context))

            logger.info(f"Job {job_id} submitted successfully for user {user_id}")
            return job_id

        except Exception as e:
            # DEBUG: [DEBUG_20251003_PDF_PARSE] - Job submission failed
            debug_context = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "component": "async_job_submission_service",
                "message": "Job submission failed",
                "context": {
                    "job_type": str(job_type),
                    "user_id": user_id or "unknown",
                    "error": str(e)
                }
            }
            logger.error(json.dumps(debug_context))

            error_msg = f"Failed to submit {str(job_type)} job: {str(e)}"
            gui_logger.log_error(e, "submit_job", context={
                "job_type": str(job_type),
                "user_id": user_id,
                "error": str(e)
            })
            raise JobSubmissionError(error_msg, job_type=str(job_type), details={"original_error": str(e)})

    def get_job_status(self, job_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a job.

        Args:
            job_id: Job ID to check
            use_cache: Whether to use cached status if available

        Returns:
            Dict containing job status information, or None if job not found
        """
        try:
            # Check cache first if enabled
            if use_cache:
                cached_status = self._get_cached_job_status(job_id)
                if cached_status:
                    return cached_status

            # Get fresh status from queue manager
            job = self.job_queue_manager.get_job(job_id)
            if not job:
                return None

            # Convert job to status dict
            status_dict = self._job_to_status_dict(job)

            # Cache the status
            self._cache_job_status(job_id, status_dict)

            return status_dict

        except Exception as e:
            logger.error(f"Error getting job status for {job_id}: {e}")
            return None

    def get_user_jobs(self, user_id: Optional[str] = None, status_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get all jobs for a specific user.

        Args:
            user_id: User ID to get jobs for (defaults to current user)
            status_filter: Optional list of statuses to filter by

        Returns:
            List of job status dictionaries
        """
        if user_id is None:
            try:
                user_id = session_state.user_id
            except:
                # Fallback for when session state is not initialized
                user_id = "default_user"

        if not user_id:
            return []

        try:
            # Get user's active jobs from session state
            user_jobs = session_state.get_user_jobs(user_id)
            job_statuses = []

            for job_info in user_jobs:
                job_id = job_info.get("job_id")
                if job_id:
                    status = self.get_job_status(job_id)
                    if status:
                        # Apply status filter if provided
                        if status_filter is None or status.get("status") in status_filter:
                            job_statuses.append(status)

            return job_statuses

        except Exception as e:
            logger.error(f"Error getting user jobs for {user_id}: {e}")
            return []

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancellation was successful
        """
        try:
            gui_logger.log_operation_start("cancel_job", context={"job_id": job_id})

            success = self.job_queue_manager.cancel_job(job_id)

            if success:
                # Update cache
                cached_status = self._get_cached_job_status(job_id)
                if cached_status:
                    cached_status["status"] = "cancelled"
                    cached_status["completed_at"] = datetime.utcnow().isoformat()
                    self._cache_job_status(job_id, cached_status)

                gui_logger.log_operation_complete("cancel_job", 0, context={"job_id": job_id})
                logger.info(f"Job {job_id} cancelled successfully")
            else:
                gui_logger.log_warning(f"Failed to cancel job {job_id}", "cancel_job", context={"job_id": job_id})

            return success

        except Exception as e:
            gui_logger.log_error(e, "cancel_job", context={"job_id": job_id})
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False

    def restart_job(self, job_id: str) -> Optional[str]:
        """
        Restart a failed or cancelled job.

        Args:
            job_id: Job ID to restart

        Returns:
            New job ID if restart was successful, None otherwise
        """
        try:
            gui_logger.log_operation_start("restart_job", context={"job_id": job_id})

            success = self.job_queue_manager.restart_job(job_id)

            if success:
                # The restart creates a new job, but we need to get the new job ID
                # For now, we'll return the same job ID assuming restart works in-place
                gui_logger.log_operation_complete("restart_job", 0, context={"job_id": job_id})
                logger.info(f"Job {job_id} restarted successfully")
                return job_id
            else:
                gui_logger.log_warning(f"Failed to restart job {job_id}", "restart_job", context={"job_id": job_id})
                return None

        except Exception as e:
            gui_logger.log_error(e, "restart_job", context={"job_id": job_id})
            logger.error(f"Error restarting job {job_id}: {e}")
            return None

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive queue statistics.

        Returns:
            Dictionary containing queue statistics
        """
        try:
            return self.job_queue_manager.get_queue_stats()
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {
                "pending_jobs": 0,
                "running_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "total_jobs": 0,
                "error": str(e)
            }

    def wait_for_job_completion(
        self,
        job_id: str,
        timeout_seconds: int = 300,
        poll_interval: float = 2.0
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete with progress feedback.

        Args:
            job_id: Job ID to wait for
            timeout_seconds: Maximum time to wait
            poll_interval: How often to check status

        Returns:
            Final job status dictionary
        """
        start_time = time.time()

        with st.spinner(f"Processing job {job_id}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            while time.time() - start_time < timeout_seconds:
                status = self.get_job_status(job_id, use_cache=False)

                if not status:
                    status_text.error(f"Job {job_id} not found")
                    return {"status": "not_found", "job_id": job_id}

                current_status = status.get("status", "unknown")
                progress = status.get("progress", 0)

                # Update progress
                progress_bar.progress(min(progress / 100.0, 1.0))

                # Update status text
                if current_status == "running":
                    status_text.info(f"Running: {status.get('current_step', 'Processing...')}")
                elif current_status == "completed":
                    progress_bar.progress(1.0)
                    status_text.success("Job completed successfully!")
                    return status
                elif current_status in ["failed", "cancelled"]:
                    progress_bar.progress(1.0)
                    status_text.error(f"Job {current_status}")
                    return status

                time.sleep(poll_interval)

            # Timeout
            progress_bar.progress(1.0)
            status_text.error(f"Job {job_id} timed out after {timeout_seconds} seconds")
            return {
                "status": "timeout",
                "job_id": job_id,
                "timeout_seconds": timeout_seconds
            }

    def _job_to_status_dict(self, job) -> Dict[str, Any]:
        """Convert a job object to a status dictionary."""
        try:
            job_type_str = job.job_type.value
        except AttributeError:
            job_type_str = str(job.job_type)

        try:
            status_str = job.status.value
        except AttributeError:
            status_str = str(job.status)

        try:
            priority_str = job.priority.value
        except AttributeError:
            priority_str = str(job.priority)

        return {
            "job_id": job.job_id,
            "job_type": job_type_str,
            "status": status_str,
            "progress": getattr(job, 'progress', 0),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "result": job.result,
            "error": job.error,
            "user_id": job.user_id,
            "current_step": getattr(job, 'metadata', {}).get('progress_message', ''),
            "priority": priority_str,
            "worker_id": getattr(job, 'worker_id', None)
        }

    def _cache_job_status(self, job_id: str, status: Optional[Dict[str, Any]] = None) -> None:
        """Cache job status in session state."""
        if status is None:
            status = self.get_job_status(job_id, use_cache=False)
            if not status:
                return

        try:
            session_state.cache_job_status(job_id, status, self._status_cache_timeout)
        except:
            # Session state not available, skip caching
            pass

    def _get_cached_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get cached job status from session state."""
        try:
            return session_state.get_cached_job_status(job_id)
        except:
            # Session state not available
            return None

    def _add_to_user_jobs(self, job_id: str, job_type: Union[str, JobType], user_id: str) -> None:
        """Add job to user's active jobs in session state."""
        try:
            job_type_str = job_type.value
        except AttributeError:
            job_type_str = str(job_type)

        job_info = {
            "job_id": job_id,
            "job_type": job_type_str,
            "submitted_at": datetime.utcnow().isoformat(),
            "user_id": user_id
        }
        try:
            session_state.add_user_job(user_id, job_info)
        except:
            # Session state not available, skip adding to user jobs
            pass

    def clear_expired_cache(self) -> None:
        """Clear expired job status cache entries."""
        try:
            session_state.clear_expired_cache()
        except:
            # Session state not available, skip clearing
            pass


# Global service instance
async_job_service = AsyncJobSubmissionService()


def submit_extraction_job(
    parameters: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
    user_id: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to submit an extraction job.

    Args:
        parameters: Job parameters
        priority: Job priority
        user_id: User ID (defaults to current session user)
        **kwargs: Additional job options

    Returns:
        Job ID
    """
    return async_job_service.submit_job(
        JobType.EXTRACTION,
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
    """
    Convenience function to submit a batch extraction job.

    Args:
        parameters: Job parameters
        priority: Job priority
        user_id: User ID (defaults to current session user)
        **kwargs: Additional job options

    Returns:
        Job ID
    """
    return async_job_service.submit_job(
        JobType.BATCH_EXTRACTION,
        parameters,
        priority=priority,
        user_id=user_id,
        **kwargs
    )


def submit_taxonomy_job(
    parameters: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
    user_id: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to submit a taxonomy processing job.

    Args:
        parameters: Job parameters
        priority: Job priority
        user_id: User ID (defaults to current session user)
        **kwargs: Additional job options

    Returns:
        Job ID
    """
    return async_job_service.submit_job(
        JobType.TAXONOMY_PROCESSING,
        parameters,
        priority=priority,
        user_id=user_id,
        **kwargs
    )


def submit_document_processing_job(
    parameters: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
    user_id: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to submit a document processing job (parsing/chunking).

    Args:
        parameters: Job parameters
        priority: Job priority
        user_id: User ID (defaults to current session user)
        **kwargs: Additional job options

    Returns:
        Job ID
    """
    return async_job_service.submit_job(
        JobType.DOCUMENT_PROCESSING,
        parameters,
        priority=priority,
        user_id=user_id,
        **kwargs
    )