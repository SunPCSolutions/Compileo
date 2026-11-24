"""
Performance optimization routes for extraction workflows.
Provides endpoints for caching, job queues, lazy loading, and cleanup management.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from datetime import datetime

from ...storage.src.database import get_tenant_db_connection
from ..core.database import get_db
from ...features.extraction.enhanced_result_search import PaginatedResultSearch, LazyResultIterator
from ...features.jobhandle.enhanced_job_queue import enhanced_job_queue_manager
from ...features.jobhandle.models import JobPriority, JobStatus
from ...features.extraction.enhanced_cache import cache_manager
from ...features.extraction.enhanced_cleanup import EnhancedCleanupService
from ...features.extraction.storage import TimeBasedRetentionPolicy
from ..core.dependencies import get_current_active_user, get_current_admin_user
from ..models.user import User as UserSchema

logger = logging.getLogger(__name__)

router = APIRouter()

# Global services (would be better with dependency injection)
_cleanup_service = None

def get_cleanup_service():
    """Get or create cleanup service singleton."""
    global _cleanup_service
    # This is a bit problematic as it needs a DB connection but we want a singleton service
    # Ideally this service should manage its own connections or take a connection factory
    if _cleanup_service is None:
        # We need a system level or admin connection here typically, or it should be instantiated per request
        # For now, let's defer instantiation or use a system connection if appropriate
        # Assuming cleanup is a system-wide or per-tenant task.
        # If it's system wide:
        from ...storage.src.database import get_system_db_connection
        # Using context manager to get a connection for initialization?
        # EnhancedCleanupService seems to expect a connection object.
        # We should likely refactor EnhancedCleanupService to take a connection factory or handle context managers.
        # For this fix, let's skip auto-initialization here and handle it per-request if possible, 
        # or use a raw connection (but ensure it's closed/managed).
        pass
    return _cleanup_service

# Dependency for paginated search
def get_paginated_search(db=Depends(get_db)):
    """Get paginated search instance."""
    return PaginatedResultSearch(db_connection=db, cache=cache_manager.get_default_cache())

# Cache management endpoints
@router.get("/cache/stats")
async def get_cache_stats():
    """Get comprehensive cache statistics."""
    try:
        return cache_manager.get_all_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@router.post("/cache/clear")
async def clear_cache():
    """Clear all caches."""
    try:
        cache_manager.cleanup_all()
        return {"message": "All caches cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear caches: {str(e)}")

# Job queue management endpoints
@router.post("/jobs/submit")
async def submit_extraction_job(
    job_type: str,
    parameters: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Submit a job to the queue for current user."""
    try:
        parameters['owner_id'] = current_user.id
        job_id = enhanced_job_queue_manager.submit_job(job_type, parameters, priority=priority, user_id=current_user.id)
        return {
            "job_id": job_id,
            "status": "submitted",
            "message": f"Job {job_id} submitted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get job status by ID for current user."""
    try:
        job = enhanced_job_queue_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        # Verify ownership
        if job.user_id != current_user.id:
             raise HTTPException(status_code=403, detail="Access denied")

        return job.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Cancel a job for current user."""
    try:
        job = enhanced_job_queue_manager.get_job(job_id)
        if job and job.user_id != current_user.id:
             raise HTTPException(status_code=403, detail="Access denied")

        success = enhanced_job_queue_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")

        return {"message": f"Job {job_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.get("/jobs/queue/stats")
async def get_queue_stats(current_user: UserSchema = Depends(get_current_admin_user)):
    """Get job queue statistics (Admin only)."""
    try:
        return enhanced_job_queue_manager.get_queue_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")

# Enhanced search endpoints
@router.get("/search/paginated")
async def search_paginated(
    query: str = "",
    categories: Optional[List[str]] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    page: int = Query(0, ge=0),
    per_page: int = Query(50, ge=1, le=200),
    search_engine=Depends(get_paginated_search),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Enhanced paginated search with caching for current user."""
    try:
        results, total_count, metadata = search_engine.search_with_pagination(
            query=query,
            categories=categories,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            page=page,
            per_page=per_page
        )

        return {
            "results": results,
            "total_count": total_count,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search/lazy/count")
async def get_lazy_search_count(
    query: str = "",
    categories: Optional[List[str]] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    search_engine=Depends(get_paginated_search),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get total count for lazy search for current user."""
    try:
        iterator = search_engine.get_lazy_iterator(
            query=query,
            categories=categories,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to
        )
        total_count = iterator.get_total_count()

        return {"total_count": total_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}")

@router.get("/search/suggestions")
async def get_search_suggestions(
    query: str,
    limit: int = Query(10, ge=1, le=50),
    search_engine=Depends(get_paginated_search),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get search suggestions based on partial query for current user."""
    try:
        suggestions = search_engine.get_search_suggestions(query, current_user.id, limit)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

# Cleanup management endpoints
# Note: Cleanup service initialization needs careful handling for multi-tenant architecture.
# For now, disabling these endpoints or mocking them if they rely on the broken global initialization.
# In a full multi-tenant system, cleanup might be a background system task rather than per-request.

@router.get("/performance/metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics."""
    try:
        metrics = {
            "cache": cache_manager.get_all_stats(),
            "jobs": enhanced_job_queue_manager.get_queue_stats(),
            # "cleanup": get_cleanup_service().get_cleanup_stats() # Disabled for now
        }

        # Add system-wide metrics
        total_cache_size = sum(
            cache_stats.get('total_size_bytes', 0)
            for cache_stats in metrics['cache'].values()
        )

        metrics['system'] = {
            'total_cache_size_bytes': total_cache_size,
            'active_jobs': metrics['jobs']['running_jobs'],
            'pending_jobs': metrics['jobs']['pending_jobs'],
            # 'cleanup_percentage': metrics['cleanup']['cleanup_percentage'] # Disabled
        }

        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.post("/performance/reset")
async def reset_performance_metrics(current_user: UserSchema = Depends(get_current_admin_user)):
    """Reset all performance metrics and caches (Admin only)."""
    try:
        # Clear caches
        cache_manager.cleanup_all()

        return {"message": "Performance metrics reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset metrics: {str(e)}")
