import sys
import os
import logging

# CRITICAL: Set up paths BEFORE any other imports to prevent import errors
# Go up 4 levels to reach the project root from src/compileo/features/jobhandle/worker.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now safe to import
from src.compileo.features.jobhandle.enhanced_job_queue import initialize_job_queue_manager, start_enhanced_worker
from src.compileo.storage.src.database import get_db_connection
from src.compileo.core.settings import BackendSettings
from src.compileo.api.core.config import settings
from src.compileo.core.logging import setup_logging, get_logger

# Configure logging for the worker
setup_logging()
logger = get_logger(__name__)

# Initialize plugin manager to ensure plugins are loaded in worker process
try:
    from src.compileo.features.plugin.manager import plugin_manager
    logger.info("WORKER: Plugin manager initialized")
except Exception as e:
    logger.error(f"WORKER: Plugin manager initialization failed: {e}")

def main():
    """
    Initializes the job queue manager and starts the RQ worker.
    This script is intended to be run as the main entry point for a worker process.
    """
    logger.info("WORKER: Worker process started.")

    # Check if another RQ worker is already running
    import subprocess
    try:
        # Check for RQ Worker processes specifically (more reliable detection)
        rq_result = subprocess.run(['pgrep', '-f', 'rq.worker'], capture_output=True, text=True)
        rq_workers = rq_result.stdout.strip().split('\n') if rq_result.stdout.strip() else []
        rq_workers = [w for w in rq_workers if w]  # Remove empty strings

        if rq_workers:
            logger.warning(f"WORKER: Found {len(rq_workers)} existing RQ worker processes: {rq_workers}")
            logger.warning("WORKER: Another RQ worker is already running. Exiting to prevent duplicate workers.")
            return
    except Exception as check_error:
        logger.warning(f"WORKER: Failed to check for existing RQ workers: {check_error}")
        logger.info("WORKER: Proceeding with worker startup anyway...")

    try:
        logger.info("WORKER: Getting database connection...")
        db_conn = get_db_connection()
        logger.info("WORKER: Database connection obtained.")

        # Get settings for job limits
        backend_settings = BackendSettings()
        global_max_jobs = backend_settings.get_global_max_concurrent_jobs()
        # TODO: Uncomment when multi-user architecture is implemented
        # per_user_max_jobs = backend_settings.get_per_user_max_concurrent_jobs()
        per_user_max_jobs = global_max_jobs  # Single-user mode
        
        logger.info(f"WORKER: Configuring job limits - Global: {global_max_jobs}, Per-User: {per_user_max_jobs}")

        logger.info("WORKER: Initializing job queue manager...")
        # Initialize with background threads disabled (API handles monitoring/cleanup)
        initialize_job_queue_manager(
            redis_url=settings.redis_url,
            db_connection=db_conn,
            global_max_jobs=global_max_jobs,
            enable_background_monitoring=False
        )
        logger.info("WORKER: Job queue manager initialized (background threads disabled).")

        logger.info("WORKER: Starting RQ worker to listen for jobs...")
        start_enhanced_worker(redis_url=settings.redis_url, default_worker_ttl=270000)  # 75 hours for long-running jobs (72h jobs + buffer)
        logger.info("WORKER: RQ worker started.")

    except Exception as e:
        logger.critical(f"WORKER: A critical error occurred during worker startup: {e}", exc_info=True)

    # DEBUG: [DEBUG_20251010_WORKER_LIFECYCLE] - Worker main function completed
    logger.info("DEBUG: [DEBUG_20251010_WORKER_LIFECYCLE] - Worker main function completed - this should not happen if worker persists")

if __name__ == "__main__":
    main()