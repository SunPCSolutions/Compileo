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
from .middleware.auth import AuthenticationMiddleware
from .routes import documents, projects, datasets, taxonomy, extraction, quality, benchmarking, prompts, dataset_versioning, jobs, chunks, plugins, settings as settings_routes
from ..core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def start_worker_process():
    """Run RQ worker in a separate process."""
    try:
        command = [sys.executable, "-m", "src.compileo.features.jobhandle.worker"]
        process = subprocess.Popen(command)
        return process
    except Exception as e:
        logger.error(f"Worker process failed: {e}", exc_info=True)
        return None


async def shutdown_rq_workers():
    """Gracefully shutdown RQ workers."""
    logger.info("Initiating RQ worker shutdown...")
    try:
        import redis
        import signal
        import os
        redis_client = redis.Redis.from_url(settings.redis_url)
        worker_pids = []
        try:
            result = subprocess.run(['pgrep', '-f', 'rq.worker'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                worker_pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
                logger.info(f"Found {len(worker_pids)} RQ worker processes: {worker_pids}")
        except Exception as e:
            logger.warning(f"Failed to find RQ worker processes: {e}")

        if not worker_pids:
            logger.info("No RQ worker processes found to shutdown")
            return

        logger.info("Sending SIGTERM to RQ workers...")
        for pid in worker_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning(f"Failed to send SIGTERM to worker PID {pid}: {e}")

        await asyncio.sleep(5)
        still_running = []
        for pid in worker_pids:
            try:
                os.kill(pid, 0)
                still_running.append(pid)
            except ProcessLookupError:
                pass

        if still_running:
            logger.warning(f"Workers still running: {still_running}")
            for pid in still_running:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.error(f"Failed to force kill worker PID {pid}: {e}")
            await asyncio.sleep(2)

        logger.info("RQ worker shutdown completed")

    except Exception as e:
        logger.error(f"Error during RQ worker shutdown: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """App initialization and cleanup."""
    logger.info("Starting Compileo API")

    from src.compileo.storage.src.project.initialize_database import setup_database
    from ..storage.src.database import get_db_connection, get_db_path
    db_file = get_db_path()

    database_needs_init = False
    if not os.path.exists(db_file):
        database_needs_init = True
    else:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gui_settings'")
            if not cursor.fetchone():
                database_needs_init = True
            conn.close()
        except Exception:
            database_needs_init = True

    if database_needs_init:
        setup_database()

    try:
        import redis
        redis_client = redis.Redis.from_url(settings.redis_url)
        redis_client.ping()
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise RuntimeError(f"Redis connection failed: {e}")

    try:
        from src.compileo.features.plugin.manager import plugin_manager
        app.state.plugin_manager = plugin_manager
    except Exception as e:
        logger.error(f"Plugin manager initialization failed: {e}")

    try:
        from src.compileo.features.jobhandle.enhanced_job_queue import EnhancedJobQueueManager
        from src.compileo.storage.src.database import get_db_connection
        db_connection = get_db_connection()
        manager = EnhancedJobQueueManager(
            redis_url=settings.redis_url,
            db_connection=db_connection,
            global_max_jobs=settings.global_max_concurrent_jobs
        )
        app.state.job_queue_manager = manager
    except Exception as e:
        logger.error(f"Job queue manager initialization failed: {e}")
        raise

    try:
        worker_process = start_worker_process()
        app.state.worker_process = worker_process
        if worker_process:
            logger.info(f"RQ worker started (PID: {worker_process.pid})")
    except Exception as e:
        logger.error(f"Failed to start RQ worker: {e}")
        app.state.worker_process = None

    yield
    logger.info("Shutting down Compileo API")
    await shutdown_rq_workers()

    if hasattr(app.state, 'worker_process') and app.state.worker_process and app.state.worker_process.poll() is None:
        app.state.worker_process.terminate()
        try:
            app.state.worker_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            app.state.worker_process.kill()


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
app.add_middleware(AuthenticationMiddleware)

# Include routers
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
    plugins.router,
    prefix="/api/v1/plugins",
    tags=["plugins"]
)

app.include_router(
    settings_routes.router,
    prefix="/api/v1/settings",
    tags=["settings"]
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
    import argparse
    from src.compileo.core.settings import BackendSettings, LogLevel
    
    parser = argparse.ArgumentParser(description="Run the Compileo API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--api-key", help="Static API key for authentication (overrides DB, supplements ENV)")
    parser.add_argument("--no-reload", action="store_false", dest="reload", help="Disable auto-reload")
    parser.set_defaults(reload=True)
    
    args = parser.parse_args()

    # If API key provided via CLI, inject it into settings
    if args.api_key:
        settings.set_cli_api_key(args.api_key)
        logger.info("API key provided via CLI argument")

    # Map our LogLevel to uvicorn log levels
    log_level_value = BackendSettings.get_log_level()
    uvicorn_log_level = "info"
    if log_level_value == LogLevel.DEBUG:
        uvicorn_log_level = "debug"
    elif log_level_value == LogLevel.ERROR:
        uvicorn_log_level = "error"
    elif log_level_value == LogLevel.NONE:
        uvicorn_log_level = "critical" # Uvicorn doesn't have 'none', critical is closest to silent
        
    uvicorn.run(
        "src.compileo.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=uvicorn_log_level
    )