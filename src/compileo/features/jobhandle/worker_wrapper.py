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
    from src.compileo.core.logging import setup_logging
    import logging
    import importlib
    
    # Initialize logging for the worker process immediately
    setup_logging()
    logger = logging.getLogger(__name__)

    # Diagnostic: correlate the RQ job id with our internal Compileo job id
    try:
        rq_module = importlib.import_module("rq")
        current_rq_job = rq_module.get_current_job()
        logger.info(
            f"DEBUG: [DEBUG_20260412_RQ_MAPPING] - worker_wrapper entry: "
            f"rq_job_id={current_rq_job.id if current_rq_job else 'none'}, "
            f"compileo_job_id={job_id}"
        )
    except Exception as rq_mapping_error:
        logger.debug(f"DEBUG: [DEBUG_20260412_RQ_MAPPING] - failed to inspect current rq job: {rq_mapping_error}")
    
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
        # Use safe defaults - ResourceManager now handles per-job two-tier limit resolution
        global_max_jobs = 10
        per_user_max_jobs = 6

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
        print("Usage: python worker_wrapper.py <job_id>")
