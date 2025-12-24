"""Jobs routes for real-time job status monitoring."""

import asyncio
import time
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime

from ...features.jobhandle.models import JobStatus

# Import enhanced_job_queue_manager dynamically to avoid import timing issues
def get_job_queue_manager():
    # Access the global variable directly at runtime
    import src.compileo.features.jobhandle.enhanced_job_queue as enhanced_job_queue
    return enhanced_job_queue.enhanced_job_queue_manager

# Function to get job queue manager from app state (fallback to global)
def get_job_queue_manager_from_app(request=None):
    """Get job queue manager from FastAPI app state or global variable."""
    # Try app state first, then global, then return None
    if request and hasattr(request.app.state, 'job_queue_manager') and request.app.state.job_queue_manager:
        return request.app.state.job_queue_manager
    global_manager = get_job_queue_manager()
    if global_manager:
        return global_manager
    return None

router = APIRouter()

# Pydantic models
class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    current_step: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime

class JobStatusUpdate(BaseModel):
    job_id: str
    previous_status: str
    new_status: str
    progress: float = 0.0
    current_step: str = ""
    timestamp: datetime

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request):
    """Get current status of a specific job."""
    try:
        # Try to get from job queue manager first
        manager = get_job_queue_manager_from_app(request)
        if manager is not None:
            job = manager.get_job(job_id)
            if job:
                return JobStatusResponse(
                    job_id=job.job_id,
                    status=job.status.value,
                    progress=job.progress,
                    current_step=job.metadata.get("progress_message", ""),
                    result=job.result,
                    error=job.error,
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                    updated_at=job.completed_at or datetime.utcnow()
                )

        # Fallback: Try to get job directly from Redis
        from ...features.jobhandle.enhanced_job_queue import EnhancedExtractionJob
        import redis
        from ...core.settings import settings

        redis_client = redis.Redis.from_url(settings.redis_url)
        job_data = redis_client.get(f"compileo:job:{job_id}")

        if job_data:
            job = EnhancedExtractionJob.from_dict(json.loads(job_data))
            return JobStatusResponse(
                job_id=job.job_id,
                status=job.status.value,
                progress=job.progress,
                current_step=job.metadata.get("progress_message", ""),
                result=job.result,
                error=job.error,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                updated_at=job.completed_at or datetime.utcnow()
            )

        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.get("/status/{job_id}/poll")
async def poll_job_status(
    job_id: str,
    request: Request,
    timeout: int = Query(30, description="Maximum time to wait for status change in seconds", ge=1, le=300),
    current_status: Optional[str] = Query(None, description="Current known status to wait for change")
):
    """
    Long poll for job status changes.
    Returns immediately if status has changed, or waits up to timeout seconds.
    """
    try:
        # Try to get from job queue manager first
        manager = get_job_queue_manager_from_app(request)
        job = None
        if manager is not None:
            job = manager.get_job(job_id)

        # Fallback: Try to get job directly from Redis
        if job is None:
            from ...features.jobhandle.enhanced_job_queue import EnhancedExtractionJob
            import redis
            from ...core.settings import settings

            redis_client = redis.Redis.from_url(settings.redis_url)
            job_data = redis_client.get(f"compileo:job:{job_id}")
            if job_data:
                job = EnhancedExtractionJob.from_dict(json.loads(job_data))

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        initial_status = job.status.value
        initial_progress = job.progress
        initial_updated_at = job.completed_at or job.created_at

        # If status has already changed from what client knows, return immediately
        if current_status and current_status != initial_status:
            return JobStatusResponse(
                job_id=job.job_id,
                status=job.status.value,
                progress=job.progress,
                current_step=job.metadata.get("progress_message", ""),
                result=job.result,
                error=job.error,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                updated_at=job.completed_at or datetime.utcnow()
            )

        # Wait for status change or timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            await asyncio.sleep(0.5)  # Check every 500ms

            if manager is not None:
                job = manager.get_job(job_id)  # Use the same manager instance
                if not job:
                    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            else:
                # Get job directly from Redis for polling
                from ...features.jobhandle.enhanced_job_queue import EnhancedExtractionJob
                import redis
                from ...core.settings import settings

                redis_client = redis.Redis.from_url(settings.redis_url)
                job_data = redis_client.get(f"compileo:job:{job_id}")
                if job_data:
                    job = EnhancedExtractionJob.from_dict(json.loads(job_data))
                else:
                    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            # Check if status, progress, or updated_at has changed
            if (job.status.value != initial_status or
                job.progress != initial_progress or
                (job.completed_at or job.created_at) != initial_updated_at):
                break

        # Return current status (whether changed or timed out)
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status.value,
            progress=job.progress,
            current_step=job.metadata.get("progress_message", ""),
            result=job.result,
            error=job.error,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            updated_at=job.completed_at or datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to poll job status: {str(e)}")

@router.get("/status/{job_id}/stream")
async def stream_job_status(job_id: str, request: Request):
    """
    Server-sent events endpoint for real-time job status updates.
    Note: This is a basic implementation. For production, consider using WebSockets.
    """
    async def generate():
        try:
            manager = get_job_queue_manager_from_app(request)
            if manager is None:
                yield f"data: {{\"error\": \"Job queue manager not initialized\"}}\n\n"
                return
            job = manager.get_job(job_id)
            if not job:
                yield f"data: {{\"error\": \"Job {job_id} not found\"}}\n\n"
                return

            last_status = job.status.value
            last_progress = job.progress
            last_updated = job.completed_at or job.created_at

            # Send initial status
            yield f"data: {{\"job_id\": \"{job_id}\", \"status\": \"{job.status.value}\", \"progress\": {job.progress}, \"current_step\": \"{job.metadata.get('progress_message', '')}\", \"timestamp\": \"{(job.completed_at or datetime.utcnow()).isoformat()}\"}}\n\n"

            # Keep connection alive and check for updates
            while True:
                await asyncio.sleep(1)  # Check every second

                job = manager.get_job(job_id)
                if not job:
                    yield f"data: {{\"error\": \"Job {job_id} not found\"}}\n\n"
                    break

                # Send update if status or progress changed
                if (job.status.value != last_status or
                    job.progress != last_progress or
                    (job.completed_at or job.created_at) != last_updated):

                    yield f"data: {{\"job_id\": \"{job_id}\", \"status\": \"{job.status.value}\", \"progress\": {job.progress}, \"current_step\": \"{job.metadata.get('progress_message', '')}\", \"timestamp\": \"{(job.completed_at or datetime.utcnow()).isoformat()}\"}}\n\n"

                    last_status = job.status.value
                    last_progress = job.progress
                    last_updated = job.completed_at or job.created_at

                    # If job is completed or failed, close the connection
                    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                        break

        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@router.get("/queue/stats")
async def get_queue_stats(request: Request):
    """Get comprehensive queue statistics."""
    try:
        manager = get_job_queue_manager_from_app(request)
        if manager is None:
            raise HTTPException(status_code=503, detail="Job queue manager not initialized. Please start the worker separately.")
        stats = manager.get_queue_stats()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")

@router.post("/cancel/{job_id}")
async def cancel_job(job_id: str, request: Request):
    """Cancel a running or pending job."""
    try:
        manager = get_job_queue_manager_from_app(request)
        if manager is None:
            raise HTTPException(status_code=503, detail="Job queue manager not initialized. Please start the worker separately.")
        success = manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")

        return {"message": f"Job {job_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.post("/restart/{job_id}")
async def restart_job(job_id: str, request: Request):
    """Restart a failed or cancelled job."""
    try:
        manager = get_job_queue_manager_from_app(request)
        if manager is None:
            raise HTTPException(status_code=503, detail="Job queue manager not initialized. Please start the worker separately.")
        success = manager.restart_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be restarted")

        return {"message": f"Job {job_id} restarted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart job: {str(e)}")