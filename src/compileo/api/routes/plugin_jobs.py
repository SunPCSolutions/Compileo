"""API routes for submitting plugin background jobs."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..core.dependencies import get_current_active_user
from ..models.user import User as UserSchema
from ...features.jobhandle.models import JobPriority, JobType


router = APIRouter()


class PluginJobSubmitRequest(BaseModel):
    """Request model for plugin background task submission."""

    executor_key: str = Field(..., description="Registered plugin executor key")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    project_id: Optional[str] = None


@router.post("/submit")
async def submit_plugin_job(
    payload: PluginJobSubmitRequest,
    request: Request,
    current_user: UserSchema = Depends(get_current_active_user),
):
    """Submit a plugin task as a background job for the current user."""
    manager = getattr(request.app.state, "job_queue_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Job queue manager not initialized")

    parameters = dict(payload.parameters or {})
    parameters["executor_key"] = payload.executor_key
    parameters["owner_id"] = current_user.id

    if payload.project_id and "project_id" not in parameters:
        parameters["project_id"] = payload.project_id

    try:
        job_id = manager.submit_job(
            job_type=JobType.PLUGIN_TASK,
            parameters=parameters,
            priority=payload.priority,
            user_id=current_user.id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to submit plugin job: {exc}")

    return {
        "job_id": job_id,
        "status": "submitted",
        "job_type": JobType.PLUGIN_TASK.value,
        "executor_key": payload.executor_key,
    }

