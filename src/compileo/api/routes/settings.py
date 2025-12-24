"""API routes for global application settings."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from ...core.settings import BackendSettings, LogLevel
from ...core.logging import setup_logging

router = APIRouter()

class LogLevelUpdate(BaseModel):
    """Model for updating the log level."""
    level: LogLevel

@router.get("/log-level")
async def get_log_level():
    """Get the current global log level."""
    return {"log_level": BackendSettings.get_log_level()}

@router.post("/log-level")
async def set_log_level(update: LogLevelUpdate):
    """Set the global log level."""
    if BackendSettings.set_log_level(update.level):
        # Update current process logging
        setup_logging(update.level)
        return {"message": f"Log level updated to {update.level.value}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update log level")
