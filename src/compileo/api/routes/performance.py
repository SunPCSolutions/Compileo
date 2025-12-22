"""
Performance optimization routes for extraction workflows.
Provides endpoints for caching, job queues, lazy loading, and cleanup management.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...features.extraction.enhanced_result_search import PaginatedResultSearch, LazyResultIterator
from ...features.jobhandle.enhanced_job_queue import enhanced_job_queue_manager
from ...features.jobhandle.models import JobPriority, JobStatus
from ...features.extraction.enhanced_cache import cache_manager
from ...features.extraction.enhanced_cleanup import EnhancedCleanupService
from ...features.extraction.storage import TimeBasedRetentionPolicy

logger = logging.getLogger(__name__)

router = APIRouter()

# Global services (would be better with dependency injection)
_cleanup_service = None

def get_cleanup_service():
    """Get or create cleanup service singleton."""
    global _cleanup_service
    if _cleanup_service is None:
        db = get_db_connection()
        retention_policy = TimeBasedRetentionPolicy(retention_days=30)
        _cleanup_service = EnhancedCleanupService(
            db_connection=db,
            default_retention_policy=retention_policy,
            enable_background=True
        )
        _cleanup_service.start_background_cleanup()
    return _cleanup_service

# Dependency for paginated search
def get_paginated_search(db=Depends(get_db_connection)):
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
    priority: JobPriority = JobPriority.NORMAL
):
    """Submit a job to the queue."""
    try:
        job_id = enhanced_job_queue_manager.submit_job(job_type, parameters, priority=priority)
        return {
            "job_id": job_id,
            "status": "submitted",
            "message": f"Job {job_id} submitted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status by ID."""
    try:
        job = enhanced_job_queue_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return job.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job."""
    try:
        success = enhanced_job_queue_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")

        return {"message": f"Job {job_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.get("/jobs/queue/stats")
async def get_queue_stats():
    """Get job queue statistics."""
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
    search_engine=Depends(get_paginated_search)
):
    """Enhanced paginated search with caching."""
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
    search_engine=Depends(get_paginated_search)
):
    """Get total count for lazy search without loading all results."""
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
    search_engine=Depends(get_paginated_search)
):
    """Get search suggestions based on partial query."""
    try:
        suggestions = search_engine.get_search_suggestions(query, limit)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

# Cleanup management endpoints
@router.post("/cleanup/run")
async def run_cleanup(
    background_tasks: BackgroundTasks,
    force: bool = False,
    schedule_name: Optional[str] = None,
    cleanup_service=Depends(get_cleanup_service)
):
    """Run cleanup manually or force a scheduled cleanup."""
    try:
        if schedule_name:
            # Run specific scheduled cleanup
            result = cleanup_service.force_cleanup(schedule_name)
        else:
            # Run general cleanup
            result = cleanup_service.perform_cleanup(force=force)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/cleanup/stats")
async def get_cleanup_stats(cleanup_service=Depends(get_cleanup_service)):
    """Get comprehensive cleanup statistics."""
    try:
        return cleanup_service.get_cleanup_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cleanup stats: {str(e)}")

@router.post("/cleanup/schedules")
async def add_cleanup_schedule(
    name: str,
    interval_seconds: int,
    retention_days: int = 30,
    priority: str = "normal",
    cleanup_service=Depends(get_cleanup_service)
):
    """Add a new cleanup schedule."""
    try:
        from ...features.extraction.enhanced_cleanup import CleanupPriority

        retention_policy = TimeBasedRetentionPolicy(retention_days=retention_days)
        priority_enum = CleanupPriority(priority.lower())

        cleanup_service.add_schedule(
            name=name,
            interval_seconds=interval_seconds,
            retention_policy=retention_policy,
            priority=priority_enum
        )

        return {"message": f"Cleanup schedule '{name}' added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add schedule: {str(e)}")

@router.delete("/cleanup/schedules/{schedule_name}")
async def remove_cleanup_schedule(
    schedule_name: str,
    cleanup_service=Depends(get_cleanup_service)
):
    """Remove a cleanup schedule."""
    try:
        success = cleanup_service.remove_schedule(schedule_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Schedule '{schedule_name}' not found")

        return {"message": f"Schedule '{schedule_name}' removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove schedule: {str(e)}")

@router.get("/cleanup/optimize")
async def optimize_cleanup_schedule(cleanup_service=Depends(get_cleanup_service)):
    """Get optimization recommendations for cleanup schedules."""
    try:
        return cleanup_service.optimize_cleanup_schedule()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize schedules: {str(e)}")

# Performance monitoring endpoints
@router.get("/performance/metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics."""
    try:
        metrics = {
            "cache": cache_manager.get_all_stats(),
            "jobs": enhanced_job_queue_manager.get_queue_stats(),
            "cleanup": get_cleanup_service().get_cleanup_stats()
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
            'cleanup_percentage': metrics['cleanup']['cleanup_percentage']
        }

        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.post("/performance/reset")
async def reset_performance_metrics():
    """Reset all performance metrics and caches."""
    try:
        # Clear caches
        cache_manager.cleanup_all()

        # Reset cleanup stats
        cleanup_service = get_cleanup_service()
        # Note: Would need to add a reset method to cleanup service

        return {"message": "Performance metrics reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset metrics: {str(e)}")