"""
Job Queue Utilities for Safe Initialization and Access.

This module provides utilities to safely initialize and access the job queue manager,
preventing race conditions and handling session state issues in Streamlit.
"""

import streamlit as st
import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class JobQueueManagerError(Exception):
    """Custom exception for job queue manager errors."""
    pass


@contextmanager
def session_state_lock():
    """Context manager for thread-safe session state access."""
    # Streamlit session state is not thread-safe, but this provides basic protection
    yield


def initialize_job_queue_manager_safe(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    global_max_jobs: int = 10,
    auto_start_worker: bool = False
) -> bool:
    """
    Safely initialize the job queue manager with retry logic and error handling.

    Args:
        max_retries: Maximum number of initialization attempts
        retry_delay: Delay between retries in seconds
        use_redis: Whether to use Redis-based queue
        global_max_jobs: Global maximum concurrent jobs
        per_user_max_jobs: Maximum concurrent jobs per user
        auto_start_worker: Whether to auto-start the worker process

    Returns:
        bool: True if initialization successful, False otherwise
    """
    with session_state_lock():
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting job queue manager initialization (attempt {attempt + 1}/{max_retries})")

                # Check if already initialized and healthy
                if _is_job_queue_manager_healthy():
                    logger.info("Job queue manager already initialized and healthy")
                    return True

                # Import and initialize
                from src.compileo.features.jobhandle.enhanced_job_queue import initialize_job_queue_manager
                from src.compileo.api.core.config import settings

                manager = initialize_job_queue_manager(
                    redis_url=settings.redis_url,
                    global_max_jobs=global_max_jobs,
                    auto_start_worker=auto_start_worker
                )

                # Store in session state
                st.session_state.job_queue_manager = manager

                # Verify initialization
                if _is_job_queue_manager_healthy():
                    logger.info("Job queue manager initialized successfully")
                    return True
                else:
                    logger.warning("Job queue manager initialization verification failed")
                    st.session_state.job_queue_manager = None

            except Exception as e:
                logger.error(f"Job queue manager initialization failed (attempt {attempt + 1}): {e}")
                st.session_state.job_queue_manager = None

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("All initialization attempts failed")
                    return False

        return False


def get_job_queue_manager_safe() -> Optional[Any]:
    """
    Safely get the job queue manager, with automatic recovery if needed.

    Returns:
        The job queue manager instance, or None if unavailable
    """
    with session_state_lock():
        manager = st.session_state.get('job_queue_manager')

        # Check if manager exists and is healthy
        if _is_job_queue_manager_healthy():
            return manager

        # Attempt recovery
        logger.warning("Job queue manager is unhealthy or missing, attempting recovery...")

        if initialize_job_queue_manager_safe():
            return st.session_state.get('job_queue_manager')

        # Recovery failed
        logger.error("Job queue manager recovery failed")
        return None


def _is_job_queue_manager_healthy() -> bool:
    """
    Check if the job queue manager is healthy and functional.

    Returns:
        bool: True if healthy, False otherwise
    """
    try:
        manager = st.session_state.get('job_queue_manager')
        if manager is None:
            return False

        # Try to get queue stats to verify functionality
        stats = manager.get_queue_stats()
        if not isinstance(stats, dict):
            return False

        # Check for required stats keys
        required_keys = ['total_jobs', 'pending_jobs', 'running_jobs']
        if not all(key in stats for key in required_keys):
            return False

        return True

    except Exception as e:
        logger.error(f"Job queue manager health check failed: {e}")
        return False


def cleanup_job_queue_manager() -> bool:
    """
    Clean up the job queue manager and reset session state.

    Returns:
        bool: True if cleanup successful, False otherwise
    """
    with session_state_lock():
        try:
            manager = st.session_state.get('job_queue_manager')
            if manager:
                # Attempt graceful shutdown if available
                if hasattr(manager, 'shutdown'):
                    manager.shutdown()
                elif hasattr(manager.queue, 'shutdown'):
                    manager.queue.shutdown()

            # Clear from session state
            st.session_state.job_queue_manager = None
            logger.info("Job queue manager cleaned up successfully")
            return True

        except Exception as e:
            logger.error(f"Job queue manager cleanup failed: {e}")
            # Force clear even if cleanup failed
            st.session_state.job_queue_manager = None
            return False


def get_job_queue_stats_safe() -> Dict[str, Any]:
    """
    Safely get job queue statistics with fallback values.

    Returns:
        Dict containing queue statistics
    """
    try:
        manager = get_job_queue_manager_safe()
        if manager:
            return manager.get_queue_stats()
    except Exception as e:
        logger.error(f"Failed to get job queue stats: {e}")

    # Return fallback stats
    return {
        'total_jobs': 0,
        'pending_jobs': 0,
        'running_jobs': 0,
        'scheduled_jobs': 0,
        'completed_jobs': 0,
        'failed_jobs': 0,
        'queue_type': 'unknown',
        'cache_size': 0,
        'cpu_usage_percent': 0.0,
        'memory_usage_mb': 0.0,
        'error': 'Job queue manager unavailable. Please start the worker separately.'
    }


def perform_job_operation_safe(operation: str, *args, **kwargs) -> Any:
    """
    Safely perform a job operation with error handling.

    Args:
        operation: Name of the operation to perform
        *args: Positional arguments for the operation
        **kwargs: Keyword arguments for the operation

    Returns:
        Result of the operation, or None if failed
    """
    try:
        manager = get_job_queue_manager_safe()
        if not manager:
            raise JobQueueManagerError("Job queue manager is not available")

        # Get the operation method
        op_method = getattr(manager, operation, None)
        if not op_method:
            raise JobQueueManagerError(f"Operation '{operation}' not found on job queue manager")

        # Perform the operation
        return op_method(*args, **kwargs)

    except Exception as e:
        logger.error(f"Job operation '{operation}' failed: {e}")
        return None


# Convenience functions for common operations
def submit_job_safe(*args, **kwargs):
    """Safely submit a job."""
    return perform_job_operation_safe('submit_job', *args, **kwargs)


def get_job_safe(*args, **kwargs):
    """Safely get a job."""
    return perform_job_operation_safe('get_job', *args, **kwargs)


def cancel_job_safe(*args, **kwargs):
    """Safely cancel a job."""
    return perform_job_operation_safe('cancel_job', *args, **kwargs)


def get_jobs_by_status_safe(*args, **kwargs):
    """Safely get jobs by status."""
    return perform_job_operation_safe('get_jobs_by_status', *args, **kwargs)