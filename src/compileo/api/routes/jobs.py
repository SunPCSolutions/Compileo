"""Jobs routes for real-time job status monitoring."""

import asyncio
import time
import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime

from ...features.jobhandle.models import JobStatus
from ..core.dependencies import get_current_active_user, get_current_admin_user
from ..models.user import User as UserSchema
from ...storage.src.project.database_repositories import JobHistoryRepository, DatasetJobRepository, ExtractionJobRepository
from ...core.settings import BackendSettings

logger = logging.getLogger(__name__)

# Import enhanced_job_queue_manager dynamically to avoid import timing issues
def get_job_queue_manager():
    # Access the global variable directly at runtime
    try:
        import src.compileo.features.jobhandle.enhanced_job_queue as enhanced_job_queue
        return enhanced_job_queue.enhanced_job_queue_manager
    except Exception as e:
        logger.error(f"DEBUG_JOBS: Failed to import enhanced_job_queue: {e}")
        return None

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

class JobHistoryEntry(BaseModel):
    """Job history entry model for API responses."""
    id: str
    owner_id: str
    project_id: Optional[str] = None
    document_id: Optional[str] = None
    job_type: str
    status: str
    parameters: Optional[Dict[str, Any]] = None
    progress: float = 0.0
    result_summary: Optional[str] = None
    error_summary: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    created_at: datetime

class JobHistoryListResponse(BaseModel):
    """Response model for job history list."""
    history: List[JobHistoryEntry]
    total_count: int

class JobStatistics(BaseModel):
    """Job statistics model."""
    total_jobs: int
    completed_count: int
    failed_count: int
    avg_duration_seconds: float


class RetentionSettings(BaseModel):
    """Job history retention settings model."""
    retention_days: int
    default_days: int = 90
    min_days: int = 1
    max_days: int = 365


class RetentionSettingsUpdate(BaseModel):
    """Update retention settings request model."""
    retention_days: int

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str, 
    request: Request,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get current status of a specific job for current user."""
    try:
        logger.info(f"DEBUG_JOBS: Getting status for job {job_id} for user {current_user.id}")
        
        # Try to get from job queue manager first
        manager = get_job_queue_manager_from_app(request)
        job = None
        if manager is not None:
            try:
                job = manager.get_job(job_id)
                if job:
                    logger.info(f"DEBUG_JOBS: Job {job_id} found in manager")
            except Exception as e:
                logger.error(f"DEBUG_JOBS: Manager get_job error: {e}")

        # Fallback: Try to get job directly from Redis
        if job is None:
            logger.info(f"DEBUG_JOBS: Job {job_id} not found in manager, trying direct Redis access")
            from ...features.jobhandle.enhanced_job_queue import EnhancedExtractionJob
            import redis
            from ..core.config import settings

            try:
                redis_client = redis.Redis.from_url(settings.redis_url)
                job_data = redis_client.get(f"compileo:job:{job_id}")

                if job_data:
                    logger.info(f"DEBUG_JOBS: Job data found in Redis for {job_id}")
                    job_dict = json.loads(job_data)
                    job = EnhancedExtractionJob.from_dict(job_dict)
                    
                    if job.user_id != current_user.id:
                        logger.warning(f"DEBUG_JOBS: Access denied for job {job_id}. Job owner: {job.user_id}, Current user: {current_user.id}")
                        raise HTTPException(status_code=403, detail="Access denied")
                else:
                    logger.warning(f"DEBUG_JOBS: Job {job_id} not found in Redis")
            except Exception as redis_e:
                logger.error(f"DEBUG_JOBS: Direct Redis access failed: {redis_e}")
                if isinstance(redis_e, HTTPException):
                    raise
                # Don't raise here, let it fall through to the final check

        if job:
             return JobStatusResponse(
                job_id=job.job_id,
                status=job.status.value if hasattr(job.status, 'value') else str(job.status),
                progress=job.progress,
                current_step=job.metadata.get("progress_message", ""),
                result=job.result,
                error=job.error,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                updated_at=job.completed_at or job.started_at or job.created_at or datetime.utcnow()
            )

        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DEBUG_JOBS: Final exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.get("/status/{job_id}/poll")
async def poll_job_status(
    job_id: str,
    request: Request,
    timeout: int = Query(30, description="Maximum time to wait for status change in seconds", ge=1, le=300),
    current_status: Optional[str] = Query(None, description="Current known status to wait for change"),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Long poll for job status changes for current user.
    Returns immediately if status has changed, or waits up to timeout seconds.
    """
    try:
        # Try to get from job queue manager first
        manager = get_job_queue_manager_from_app(request)
        job = None
        if manager is not None:
            job = manager.get_job(job_id)
            if job and job.user_id != current_user.id:
                 raise HTTPException(status_code=403, detail="Access denied")

        # Fallback: Try to get job directly from Redis
        if job is None:
            from ...features.jobhandle.enhanced_job_queue import EnhancedExtractionJob
            import redis
            from ..core.config import settings

            redis_client = redis.Redis.from_url(settings.redis_url)
            job_data = redis_client.get(f"compileo:job:{job_id}")
            if job_data:
                job = EnhancedExtractionJob.from_dict(json.loads(job_data))
                if job.user_id != current_user.id:
                    raise HTTPException(status_code=403, detail="Access denied")

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        initial_status = job.status.value if hasattr(job.status, 'value') else str(job.status)
        initial_progress = job.progress
        initial_updated_at = job.completed_at or job.created_at

        # If status has already changed from what client knows, return immediately
        if current_status and current_status != initial_status:
            return JobStatusResponse(
                job_id=job.job_id,
                status=initial_status,
                progress=job.progress,
                current_step=job.metadata.get("progress_message", ""),
                result=job.result,
                error=job.error,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                updated_at=job.completed_at or job.started_at or job.created_at or datetime.utcnow()
            )

        # Wait for status change or timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            await asyncio.sleep(0.5)  # Check every 500ms

            if manager is not None:
                job = manager.get_job(job_id)
                if not job:
                    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
                if job.user_id != current_user.id:
                    raise HTTPException(status_code=403, detail="Access denied")
            else:
                # Get job directly from Redis for polling
                from ...features.jobhandle.enhanced_job_queue import EnhancedExtractionJob
                import redis
                from ..core.config import settings

                redis_client = redis.Redis.from_url(settings.redis_url)
                job_data = redis_client.get(f"compileo:job:{job_id}")
                if job_data:
                    job = EnhancedExtractionJob.from_dict(json.loads(job_data))
                    if job.user_id != current_user.id:
                        raise HTTPException(status_code=403, detail="Access denied")
                else:
                    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            # Check if status, progress, or updated_at has changed
            current_status_val = job.status.value if hasattr(job.status, 'value') else str(job.status)
            if (current_status_val != initial_status or
                job.progress != initial_progress or
                (job.completed_at or job.created_at) != initial_updated_at):
                break

        # Return current status (whether changed or timed out)
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status.value if hasattr(job.status, 'value') else str(job.status),
            progress=job.progress,
            current_step=job.metadata.get("progress_message", ""),
            result=job.result,
            error=job.error,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            updated_at=job.completed_at or job.started_at or job.created_at or datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DEBUG_JOBS: Poll exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to poll job status: {str(e)}")

@router.get("/status/{job_id}/stream")
async def stream_job_status(
    job_id: str, 
    request: Request,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Server-sent events endpoint for real-time job status updates for current user.
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
                
            # Verify ownership
            if job.user_id != current_user.id:
                yield f"data: {{\"error\": \"Access denied\"}}\n\n"
                return

            last_status = job.status.value if hasattr(job.status, 'value') else str(job.status)
            last_progress = job.progress
            last_updated = job.completed_at or job.created_at

            # Send initial status
            yield f"data: {{\"job_id\": \"{job_id}\", \"status\": \"{last_status}\", \"progress\": {job.progress}, \"current_step\": \"{job.metadata.get('progress_message', '')}\", \"timestamp\": \"{(job.completed_at or datetime.utcnow()).isoformat()}\"}}\n\n"

            # Keep connection alive and check for updates
            while True:
                await asyncio.sleep(1)  # Check every second

                job = manager.get_job(job_id)
                if not job:
                    yield f"data: {{\"error\": \"Job {job_id} not found\"}}\n\n"
                    break

                current_status_val = job.status.value if hasattr(job.status, 'value') else str(job.status)
                # Send update if status or progress changed
                if (current_status_val != last_status or
                    job.progress != last_progress or
                    (job.completed_at or job.created_at) != last_updated):

                    yield f"data: {{\"job_id\": \"{job_id}\", \"status\": \"{current_status_val}\", \"progress\": {job.progress}, \"current_step\": \"{job.metadata.get('progress_message', '')}\", \"timestamp\": \"{(job.completed_at or datetime.utcnow()).isoformat()}\"}}\n\n"

                    last_status = current_status_val
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
async def get_queue_stats(
    request: Request,
    current_user: UserSchema = Depends(get_current_admin_user)
):
    """Get comprehensive queue statistics (Admin only)."""
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
async def cancel_job(
    job_id: str, 
    request: Request,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Cancel a running or pending job for current user."""
    try:
        manager = get_job_queue_manager_from_app(request)
        if manager is None:
            raise HTTPException(status_code=503, detail="Job queue manager not initialized. Please start the worker separately.")
        
        job = manager.get_job(job_id)
        if job and job.user_id != current_user.id:
             raise HTTPException(status_code=403, detail="Access denied")

        success = manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")

        return {"message": f"Job {job_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.post("/restart/{job_id}")
async def restart_job(
    job_id: str, 
    request: Request,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Restart a failed or cancelled job for current user."""
    try:
        manager = get_job_queue_manager_from_app(request)
        if manager is None:
            raise HTTPException(status_code=503, detail="Job queue manager not initialized. Please start the worker separately.")
        
        job = manager.get_job(job_id)
        if job and job.user_id != current_user.id:
             raise HTTPException(status_code=403, detail="Access denied")

        success = manager.restart_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be restarted")

        return {"message": f"Job {job_id} restarted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart job: {str(e)}")


# Job History Endpoints

@router.get("/history", response_model=JobHistoryListResponse)
async def get_job_history(
    project_id: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Get job history for the current user.
    
    Optionally filter by project_id or job_type.
    Returns completed/failed job metadata for audit and tracking purposes.
    """
    try:
        history_repo = JobHistoryRepository(current_user.id)
        
        if project_id:
            history = history_repo.get_history_by_project(project_id, limit)
        elif job_type:
            history = history_repo.get_history_by_job_type(job_type, limit)
        else:
            # Get all history for user
            history = history_repo.get_all_history(limit)
        
        # Parse parameters JSON for each entry
        for entry in history:
            if entry.get('parameters'):
                try:
                    entry['parameters'] = json.loads(entry['parameters'])
                except:
                    entry['parameters'] = {}
        
        return JobHistoryListResponse(
            history=history,
            total_count=len(history)
        )
    except Exception as e:
        logger.error(f"Failed to get job history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job history: {str(e)}")


@router.get("/history/{history_id}", response_model=JobHistoryEntry)
async def get_job_history_entry(
    history_id: str,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get a specific job history entry by ID."""
    try:
        history_repo = JobHistoryRepository(current_user.id)
        entry = history_repo.get_history_entry(history_id)
        
        if not entry:
            raise HTTPException(status_code=404, detail=f"History entry {history_id} not found")
        
        # Parse parameters JSON
        if entry.get('parameters'):
            try:
                entry['parameters'] = json.loads(entry['parameters'])
            except:
                entry['parameters'] = {}
        
        return JobHistoryEntry(**entry)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job history entry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job history entry: {str(e)}")


@router.get("/history/statistics/summary", response_model=JobStatistics)
async def get_job_statistics(
    project_id: Optional[str] = None,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Get job statistics for the current user.
    
    Returns aggregated statistics about completed/failed jobs including:
    - Total jobs archived
    - Completion count
    - Failure count
    - Average duration
    """
    try:
        history_repo = JobHistoryRepository(current_user.id)
        stats = history_repo.get_job_statistics(project_id)
        
        return JobStatistics(**stats)
    except Exception as e:
        logger.error(f"Failed to get job statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job statistics: {str(e)}")


@router.post("/history/archive/{job_id}")
async def archive_job(
    job_id: str,
    job_type: str = Query(..., description="Type of job: dataset, extraction, benchmark, quality"),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Manually archive a completed or failed job to history.
    
    This moves the job from the active jobs table to the history table,
    retaining only metadata for audit purposes.
    """
    try:
        history_repo = JobHistoryRepository(current_user.id)
        
        # Get the appropriate job repository based on type
        if job_type == "dataset":
            job_repo = DatasetJobRepository(current_user.id)
        elif job_type == "extraction":
            job_repo = ExtractionJobRepository(current_user.id)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported job type: {job_type}")
        
        # Archive the job
        history_id = job_repo.archive_completed_job(job_id, history_repo)
        
        if not history_id:
            raise HTTPException(
                status_code=400, 
                detail=f"Job {job_id} not found or not in completed/failed state"
            )
        
        return {
            "message": f"Job {job_id} archived successfully",
            "history_id": history_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to archive job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to archive job: {str(e)}")


@router.delete("/history/cleanup")
async def cleanup_old_history(
    days_to_keep: int = Query(90, ge=1, le=365),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Clean up old job history entries.
    
    Removes history entries older than the specified number of days.
    Default retention is 90 days.
    """
    try:
        history_repo = JobHistoryRepository(current_user.id)
        deleted_count = history_repo.cleanup_old_history(days_to_keep)
        
        return {
            "message": f"Cleaned up {deleted_count} old history entries",
            "deleted_count": deleted_count,
            "retention_days": days_to_keep
        }
    except Exception as e:
        logger.error(f"Failed to cleanup job history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cleanup job history: {str(e)}")


@router.get("/history/settings/retention", response_model=RetentionSettings)
async def get_retention_settings(
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Get the current job history retention settings.
    
    Returns the configured retention period in days.
    """
    try:
        settings = BackendSettings(owner_id=current_user.id)
        retention_days = settings.get_job_history_retention_days()
        
        return RetentionSettings(
            retention_days=retention_days,
            default_days=90,
            min_days=1,
            max_days=365
        )
    except Exception as e:
        logger.error(f"Failed to get retention settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get retention settings: {str(e)}")


@router.put("/history/settings/retention", response_model=RetentionSettings)
async def update_retention_settings(
    settings_update: RetentionSettingsUpdate,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Update the job history retention settings.
    
    Sets the number of days to retain job history entries.
    Must be between 1 and 365 days.
    """
    try:
        days = settings_update.retention_days
        
        # Validate range
        if days < 1 or days > 365:
            raise HTTPException(
                status_code=400,
                detail=f"Retention days must be between 1 and 365, got {days}"
            )
        
        settings = BackendSettings(owner_id=current_user.id)
        success = settings.set_job_history_retention_days(days)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save retention settings"
            )
        
        return RetentionSettings(
            retention_days=days,
            default_days=90,
            min_days=1,
            max_days=365
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update retention settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update retention settings: {str(e)}")


@router.post("/history/cleanup/apply-settings")
async def cleanup_with_user_settings(
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Clean up old job history entries using the user's configured retention setting.
    
    This endpoint uses the retention_days setting from user preferences.
    """
    try:
        settings = BackendSettings(owner_id=current_user.id)
        days_to_keep = settings.get_job_history_retention_days()
        
        history_repo = JobHistoryRepository(current_user.id)
        deleted_count = history_repo.cleanup_old_history(days_to_keep)
        
        return {
            "message": f"Cleaned up {deleted_count} old history entries",
            "deleted_count": deleted_count,
            "retention_days": days_to_keep,
            "used_user_setting": True
        }
    except Exception as e:
        logger.error(f"Failed to cleanup job history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cleanup job history: {str(e)}")
