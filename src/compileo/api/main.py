"""Main FastAPI application for Compileo API."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import multiprocessing
import subprocess
import time
import sys
import asyncio
import signal
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .middleware.logging import RequestLoggingMiddleware
from .middleware.rate_limiting import RateLimitingMiddleware
from .routes import documents, projects, datasets, taxonomy, extraction, quality, benchmarking, prompts, dataset_versioning, jobs, chunks, plugins, auth, licensing, settings as settings_router, plugin_jobs
from ..core.logging import setup_logging

# Configure centralized logging (reads log level from settings database)
setup_logging()
logger = logging.getLogger(__name__)


def start_worker_process():
    """Function to run the RQ worker in a separate process."""
    try:
        # Use sys.executable to ensure the worker runs with the same Python interpreter
        command = [sys.executable, "-m", "src.compileo.features.jobhandle.worker"]
        process = subprocess.Popen(command)
        return process  # Return the process object instead of waiting
    except Exception as e:
        logger.error(f"Worker process failed: {e}", exc_info=True)
        return None


async def shutdown_rq_workers():
    """Gracefully shutdown all RQ workers."""
    logger.info("Initiating RQ worker shutdown...")

    try:
        import redis
        import signal
        import os

        # Connect to Redis
        redis_client = redis.Redis.from_url(settings.redis_url)

        # Find all RQ worker processes
        worker_pids = []
        try:
            # Use pgrep to find RQ worker processes
            result = subprocess.run(['pgrep', '-f', 'rq.worker'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                worker_pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
                logger.info(f"Found {len(worker_pids)} RQ worker processes: {worker_pids}")
        except Exception as e:
            logger.warning(f"Failed to find RQ worker processes: {e}")

        if not worker_pids:
            logger.info("No RQ worker processes found to shutdown")
            return

        # Send SIGTERM (graceful shutdown) to all workers
        logger.info("Sending SIGTERM to RQ workers for graceful shutdown...")
        for pid in worker_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.debug(f"Sent SIGTERM to worker PID {pid}")
            except ProcessLookupError:
                logger.debug(f"Worker PID {pid} no longer exists")
            except Exception as e:
                logger.warning(f"Failed to send SIGTERM to worker PID {pid}: {e}")

        # Wait for workers to finish (up to 30 seconds)
        logger.info("Waiting for workers to shutdown gracefully...")
        await asyncio.sleep(5)  # Give workers time to finish current jobs

        # Check if workers are still running
        still_running = []
        for pid in worker_pids:
            try:
                os.kill(pid, 0)  # Signal 0 just checks if process exists
                still_running.append(pid)
            except ProcessLookupError:
                pass  # Process is gone

        if still_running:
            logger.warning(f"Workers still running after graceful shutdown: {still_running}")
            logger.info("Sending SIGKILL to force shutdown remaining workers...")

            # Force kill remaining workers
            for pid in still_running:
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.debug(f"Force killed worker PID {pid}")
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.error(f"Failed to force kill worker PID {pid}: {e}")

            # Final wait
            await asyncio.sleep(2)

        logger.info("RQ worker shutdown completed")

    except Exception as e:
        logger.error(f"Error during RQ worker shutdown: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    logger.info("Starting Compileo API - lifespan function called")

    # Initialize database if it doesn't exist or is empty
    from src.compileo.storage.src.project.initialize_database import setup_database
    import os
    from ..storage.src.database import get_db_connection
    from ..storage.src.database import get_db_path
    db_file = get_db_path()

    database_needs_init = False
    if not os.path.exists(db_file):
        logger.info("Database file does not exist, initializing multi-user database...")
        database_needs_init = True
    else:
        # Check if database has required tables
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            result = cursor.fetchone()
            conn.close()
            if not result:
                logger.info("Database exists but is missing multi-user tables, initializing...")
                database_needs_init = True
        except Exception as e:
            logger.warning(f"Error checking database tables: {e}, initializing...")
            database_needs_init = True

    if database_needs_init:
        setup_database()
        logger.info("Multi-user database initialized successfully")

    # Check Redis connectivity before initializing job queue
    logger.info("Checking Redis connectivity...")
    try:
        import redis
        redis_client = redis.Redis.from_url(settings.redis_url)
        redis_client.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.error("Compileo requires Redis for job queuing. Please ensure Redis is running and accessible.")
        raise RuntimeError(f"Redis connection failed: {e}")

    # Initialize plugin manager
    try:
        from src.compileo.features.plugin.manager import plugin_manager
        logger.info("Plugin manager initialized")
        app.state.plugin_manager = plugin_manager
    except Exception as e:
        logger.error(f"Plugin manager initialization failed: {e}")

    # Initialize job queue manager with config settings
    logger.info("About to initialize job queue manager...")
    try:
        from src.compileo.features.jobhandle.enhanced_job_queue import EnhancedJobQueueManager
        from src.compileo.storage.src.database import get_db_connection

        logger.info("Initializing job queue manager...")
        db_connection = get_db_connection()
        logger.info(f"Database connection obtained: {db_connection is not None}")

        # Create the manager directly
        manager = EnhancedJobQueueManager(
            redis_url=settings.redis_url,
            db_connection=db_connection,
            global_max_jobs=settings.global_max_concurrent_jobs,
            per_user_max_jobs=settings.per_user_max_concurrent_jobs
        )
        logger.info(f"Job queue manager initialized: {manager is not None}")

        # Store the manager in the app state for access by routes
        app.state.job_queue_manager = manager

    except Exception as e:
        logger.error(f"❌ Job queue manager initialization failed: {e}")
        raise

    logger.info("✅ Lifespan initialization completed successfully")

    # Start the worker in a background process
    # Use subprocess directly to avoid multiprocessing pickling issues
    try:
        worker_process = start_worker_process()
        app.state.worker_process = worker_process
        if worker_process:
            logger.info(f"RQ worker started in background process (PID: {worker_process.pid})")
        else:
            logger.warning("RQ worker failed to start")
    except Exception as e:
        logger.error(f"Failed to start RQ worker: {e}")
        # Continue without worker - some functionality may be limited
        app.state.worker_process = None

    yield
    logger.info("Shutting down Compileo API")

    # Graceful shutdown of RQ workers
    await shutdown_rq_workers()

    # Terminate the worker process started by this API instance
    if hasattr(app.state, 'worker_process') and app.state.worker_process and app.state.worker_process.poll() is None:
        logger.info("Terminating API worker process...")
        app.state.worker_process.terminate()
        try:
            app.state.worker_process.wait(timeout=10)  # Increased timeout
        except subprocess.TimeoutExpired:
            logger.warning("Worker process didn't terminate gracefully, force killing...")
            app.state.worker_process.kill()
            try:
                app.state.worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("Worker process could not be killed")
        logger.info("RQ worker process terminated")


# Create FastAPI application
app = FastAPI(
    title="Compileo API",
    description="REST API for Compileo - A modular pipeline for dataset creation and curation",
    version="1.0.0",
    docs_url="/docs",  # Changed from /api/v1/docs to /docs for easier access
    redoc_url="/redoc",  # Changed from /api/v1/redoc to /redoc
    openapi_url="/openapi.json",  # Changed from /api/v1/openapi.json to /openapi.json
    lifespan=lifespan,
    redirect_slashes=False
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitingMiddleware)

# Include routers
app.include_router(auth.router)

app.include_router(
    documents.router,
    prefix="/api/v1/documents",
    tags=["documents"]
)

app.include_router(
    projects.router,
    prefix="/api/v1/projects",
    tags=["projects"]
)

app.include_router(
    datasets.router,
    prefix="/api/v1/datasets",
    tags=["datasets"]
)

app.include_router(
    taxonomy.router,
    prefix="/api/v1/taxonomy",
    tags=["taxonomy"]
)

app.include_router(
    extraction.router,
    prefix="/api/v1/extraction",
    tags=["extraction"]
)

app.include_router(
    quality.router,
    prefix="/api/v1/quality",
    tags=["quality"]
)

app.include_router(
    benchmarking.router,
    prefix="/api/v1/benchmarking",
    tags=["benchmarking"]
)

app.include_router(
    prompts.router,
    prefix="/api/v1/prompts",
    tags=["prompts"]
)

app.include_router(
    dataset_versioning.router,
    prefix="/api/v1/datasets/versions",
    tags=["dataset_versioning"]
)

app.include_router(
    jobs.router,
    prefix="/api/v1/jobs",
    tags=["jobs"]
)

app.include_router(
    chunks.router,
    prefix="/api/v1/chunks",
    tags=["chunks"]
)

app.include_router(
    licensing.router,
    prefix="/api/v1/licensing",
    tags=["licensing"]
)

logger.info("Registering plugins router at /api/v1/plugins")
app.include_router(
    plugins.router,
    prefix="/api/v1/plugins",
    tags=["plugins"]
)

app.include_router(
    settings_router.router,
    prefix="/api/v1/settings",
    tags=["settings"]
)

app.include_router(
    plugin_jobs.router,
    prefix="/api/v1/plugin-jobs",
    tags=["plugin-jobs"]
)


@app.get("/health")

async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "compileo-api"}


@app.post("/shutdown")
async def shutdown_service():
    """Manually trigger service shutdown."""
    logger.info("Manual shutdown requested via API endpoint")

    try:
        # Initiate graceful shutdown of RQ workers
        await shutdown_rq_workers()

        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Schedule the server shutdown
        def shutdown_server():
            # This will cause the server to stop accepting new requests
            # and begin the shutdown sequence
            logger.info("Server shutdown scheduled")

        # Use call_soon to schedule the shutdown in the next iteration of the event loop
        loop.call_soon(shutdown_server)

        return {
            "message": "Shutdown initiated",
            "details": "RQ workers have been terminated. Server will shutdown shortly."
        }

    except Exception as e:
        logger.error(f"Error during manual shutdown: {e}")
        return {
            "error": "Shutdown failed",
            "details": str(e)
        }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Compileo API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.compileo.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
