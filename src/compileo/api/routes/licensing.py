from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from ...features.licensing.manager import LicenseManager
from ...features.licensing.models import LicenseStatus
from ..core.dependencies import get_current_active_user
from ..models.user import User

import logging

logger = logging.getLogger(__name__)

router = APIRouter()
license_manager = LicenseManager()

@router.get("/status", response_model=Optional[LicenseStatus])
async def get_license_status(current_user: User = Depends(get_current_active_user)):
    """Gets the current license status."""
    try:
        status = license_manager.get_local_status(current_user.id)
        logger.info(f"License status for {current_user.id}: {status}")
        return status
    except Exception as e:
        logger.error(f"Error getting license status for {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/activate")
async def activate_license(key: str, current_user: User = Depends(get_current_active_user)):
    """Activates a license key."""
    success, message = await license_manager.activate(key, current_user.id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

@router.post("/validate")
async def validate_license(current_user: User = Depends(get_current_active_user)):
    """Manually triggers a remote license validation."""
    is_valid = await license_manager.validate(current_user.id, force=True)
    return {"is_valid": is_valid}
