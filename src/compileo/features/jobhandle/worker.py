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
from src.compileo.core.logging import setup_logging

# Configure centralized logging (reads log level from settings database)
setup_logging()
logger = logging.getLogger(__name__)

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

    # NOTE: Removed the pgrep check that was preventing multiple workers in Docker.
    # In Docker environments, each container is isolated, so we don't need to check
    # for other workers on the system. The worker deduplication logic was incorrectly
    # detecting workers from other containers and exiting.
    #
    # Docker handles worker isolation through containerization.
    # Horizontal scaling via WORKER_REPLICAS is now fully supported.

    try:
        logger.info("WORKER: Getting database connection...")
        db_conn = get_db_connection()
        logger.info("WORKER: Database connection obtained.")

        # Use safe defaults - ResourceManager now handles per-job two-tier limit resolution
        global_max_jobs = 10
        per_user_max_jobs = 6

        logger.info(f"WORKER: Configuring job limits - Global: {global_max_jobs}, Per-User: {per_user_max_jobs}")

        logger.info("WORKER: Initializing job queue manager...")
        # Initialize with background threads disabled (API handles monitoring/cleanup)
        initialize_job_queue_manager(
            redis_url=settings.redis_url,
            db_connection=db_conn,
            global_max_jobs=global_max_jobs,
            per_user_max_jobs=per_user_max_jobs,
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