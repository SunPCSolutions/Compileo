#!/usr/bin/env python3
"""
Simple RQ worker wrapper that can be called by RQ and delegates to the actual job processing.
This avoids RQ serialization issues with complex function objects.
"""

import sys
import os

# Add the project root to Python path - Go up 4 levels to reach root from src/compileo/features/jobhandle/worker_wrapper.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
src_root = os.path.join(project_root, 'src')

# Add both project root and src directory to ensure imports work
if src_root not in sys.path:
    sys.path.insert(0, src_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add to environment for subprocesses
os.environ['PYTHONPATH'] = f"{src_root}:{project_root}:{os.environ.get('PYTHONPATH', '')}"

def process_job(job_id: str, **kwargs) -> None:
    """Process a job by ID - this function can be called by RQ."""
    import sys
    from ...core.logging import setup_logging, get_logger
    
    # Initialize logging for the worker process
    setup_logging()
    logger = get_logger(__name__)
    
    logger.debug(f"WORKER_WRAPPER_CALLED_V2: process_job called with job_id={job_id}")
    try:
        # Initialize job queue manager in worker process
        # IMPORTANT: Disable background monitoring and cleanup threads in worker wrapper
        # The main API process and QueueProcessor already handle cleanup and monitoring.
        # Running them in every short-lived worker wrapper is redundant and dangerous (race conditions).
        from .enhanced_job_queue import initialize_job_queue_manager
        
        # Get settings for limits and Redis URL
        from src.compileo.core.settings import BackendSettings
        from src.compileo.api.core.config import settings
        backend_settings = BackendSettings()
        global_max_jobs = backend_settings.get_global_max_concurrent_jobs()
        # TODO: Uncomment when multi-user architecture is implemented
        # per_user_max_jobs = backend_settings.get_per_user_max_concurrent_jobs()
        per_user_max_jobs = global_max_jobs  # Single-user mode

        # Initialize WITHOUT background monitoring
        initialize_job_queue_manager(
            redis_url=settings.redis_url,
            global_max_jobs=global_max_jobs,
            enable_background_monitoring=False,
            auto_start_worker=False
        )

        # Import the actual job processing logic
        from .enhanced_job_queue import redis_job_wrapper

        # Call the actual job processor (ignore extra RQ kwargs)
        logger.debug(f"WORKER_WRAPPER_BEFORE_REDIS: About to call redis_job_wrapper for {job_id}")
        result = redis_job_wrapper(job_id)
        logger.debug(f"WORKER_WRAPPER_SUCCESS: redis_job_wrapper returned for {job_id}")
        return result

    except Exception as e:
        logger.error(f"Error in process_job: {e}")
        raise

if __name__ == "__main__":
    # Allow direct execution for testing
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
        process_job(job_id)
    else:
        # We can use standard print for usage
        print("Usage: python worker_wrapper.py <job_id>")