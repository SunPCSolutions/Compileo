"""Benchmarking routes for the Compileo API."""

import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...features.jobhandle.enhanced_job_queue import enhanced_job_queue_manager, JobType, JobPriority

# Create router
router = APIRouter()

# Pydantic models
class BenchmarkRunRequest(BaseModel):
    model_info: dict
    evaluation_results: dict
    suite: str = "glue"
    config: Optional[dict] = None

class BenchmarkResult(BaseModel):
    job_id: str
    status: str
    summary: dict
    performance_data: dict
    comparisons: dict
    completed_at: Optional[datetime] = None

class BenchmarkComparisonRequest(BaseModel):
    model_ids: list
    benchmark_suite: str = "glue"
    metrics: Optional[list] = None

# In-memory storage for benchmark jobs (replace with proper storage in production)
benchmark_jobs = {}

# Dependency to get database connection
def get_db():
    return get_db_connection()

@router.post("/run")
async def run_benchmarks(
    request: BenchmarkRunRequest,
    db=Depends(get_db)
):
    """Run AI model benchmarks."""
    try:
        # Prepare job parameters
        job_params = {
            "job_id": str(uuid.uuid4()),  # Will be overridden by job queue
            "project_id": request.model_info.get("project_id", 1),  # Default project
            "benchmark_suite": request.suite,
            "ai_config": request.config or {},
            "benchmark_params": {}
        }

        # Submit job to queue
        job_id = enhanced_job_queue_manager.submit_job(
            job_type=JobType.BENCHMARKING,  # Assuming this exists or add it
            parameters=job_params,
            priority=JobPriority.NORMAL,
            user_id=request.model_info.get("user_id")
        )

        return {
            "job_id": job_id,
            "message": f"Benchmarking started for suite: {request.suite}",
            "estimated_duration": "10-30 minutes"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start benchmarking: {str(e)}")

@router.get("/results/{job_id}", response_model=BenchmarkResult)
async def get_benchmark_results(job_id: str):
    """Get benchmarking results."""
    # Try to get from database first via repository pattern
    # For now, we'll use the mock/in-memory fallback if not implemented fully
    if job_id not in benchmark_jobs:
        # Check if it exists in the job queue manager
        job = enhanced_job_queue_manager.get_job(job_id)
        if job:
            # Construct a result object from the job status
            return BenchmarkResult(
                job_id=job.job_id,
                status=job.status.value,
                summary={},
                performance_data={},
                comparisons={},
                completed_at=job.completed_at
            )
        raise HTTPException(status_code=404, detail=f"Benchmark job {job_id} not found")

    return benchmark_jobs[job_id]

@router.post("/cancel/{job_id}")
async def cancel_benchmark_job(job_id: str):
    """Cancel a running benchmark job."""
    try:
        success = enhanced_job_queue_manager.cancel_job(job_id)
        if success:
            return {"message": f"Job {job_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to cancel job {job_id} or job not found/already completed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")

@router.get("/results")
async def list_benchmark_results(
    model_name: Optional[str] = None,
    suite: Optional[str] = None,
    limit: int = 20,
    db=Depends(get_db)
):
    """List benchmark results with filtering."""
    try:
        # TODO: Implement proper results storage and retrieval
        # For now, return mock data
        results = []
        for job in list(benchmark_jobs.values())[:limit]:
            if job.status == "completed":
                results.append(job)

        return {"results": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list benchmark results: {str(e)}")

@router.post("/compare")
async def compare_models(request: BenchmarkComparisonRequest, db=Depends(get_db)):
    """Compare performance across models."""
    try:
        # TODO: Implement model comparison logic
        # For now, return mock comparison
        return {
            "comparison": {
                "models_compared": request.model_ids,
                "suite": request.benchmark_suite,
                "metrics": ["accuracy", "f1_score", "precision", "recall"],
                "results": {
                    "best_performing": request.model_ids[0] if request.model_ids else None,
                    "performance_gap": 0.05,
                    "recommendations": ["Consider fine-tuning on domain-specific data"]
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")

@router.get("/history")
async def get_benchmark_history(
    model_name: Optional[str] = None,
    days: int = 30,
    db=Depends(get_db)
):
    """Get benchmarking history."""
    try:
        # TODO: Implement proper history storage
        # For now, return mock data
        return {
            "history": list(benchmark_jobs.values()),
            "total_runs": len(benchmark_jobs),
            "date_range": f"Last {days} days"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get benchmark history: {str(e)}")

@router.get("/leaderboard")
async def get_leaderboard(
    suite: str = "glue",
    metric: str = "accuracy",
    limit: int = 10,
    db=Depends(get_db)
):
    """Get model leaderboard for a benchmark suite."""
    try:
        # TODO: Implement leaderboard logic
        # For now, return mock leaderboard
        return {
            "suite": suite,
            "metric": metric,
            "leaderboard": [
                {"rank": 1, "model": "gpt-4", "score": 0.95},
                {"rank": 2, "model": "claude-3", "score": 0.92},
                {"rank": 3, "model": "gemini-pro", "score": 0.89}
            ],
            "total_models": 150
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {str(e)}")

# Background task function
def process_benchmarking(job_id: str, request: BenchmarkRunRequest, db):
    """Background task for benchmarking."""
    try:
        import time

        # Simulate benchmarking progress
        time.sleep(2)
        benchmark_jobs[job_id].status = "running"

        # TODO: Implement actual benchmarking using BenchmarkingModule
        # For now, provide mock results
        time.sleep(5)

        mock_summary = {
            "total_evaluations": 5,
            "benchmarks_run": [request.suite],
            "models_evaluated": 1,
            "total_time_seconds": 45.2
        }

        mock_performance = {
            request.suite: {
                "accuracy": {"mean": 0.87, "std": 0.02},
                "f1_score": {"mean": 0.85, "std": 0.03},
                "precision": {"mean": 0.88, "std": 0.02},
                "recall": {"mean": 0.83, "std": 0.04}
            }
        }

        mock_comparisons = {
            "baseline_comparison": {
                "improvement": 0.05,
                "statistical_significance": "p < 0.01"
            }
        }

        benchmark_jobs[job_id].status = "completed"
        benchmark_jobs[job_id].summary = mock_summary
        benchmark_jobs[job_id].performance_data = mock_performance
        benchmark_jobs[job_id].comparisons = mock_comparisons
        benchmark_jobs[job_id].completed_at = datetime.utcnow()

    except Exception as e:
        benchmark_jobs[job_id].status = "failed"
        benchmark_jobs[job_id].error = str(e)
