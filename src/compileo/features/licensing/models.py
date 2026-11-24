from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class LicenseDetails(BaseModel):
    id: int
    license_key: str
    expires_at: Optional[datetime] = None
    status: int
    is_expired: bool
    times_activated: int
    activations_limit: Optional[int] = None
    user_id: Optional[int] = None
    product_id: Optional[int] = None

class ActivationData(BaseModel):
    id: int
    token: str
    license_id: Optional[int] = None
    label: Optional[str] = None
    created_at: datetime
    deactivated_at: Optional[datetime] = None
    license: Optional[LicenseDetails] = None

class ActivationResponse(BaseModel):
    success: bool = False
    data: Optional[ActivationData] = None
    code: Optional[str] = None
    message: Optional[str] = None

class ValidationResponse(BaseModel):
    success: bool = False
    data: Optional[ActivationData] = None
    code: Optional[str] = None
    message: Optional[str] = None

class LicenseStatus(BaseModel):
    key: str
    status: str  # 'active', 'inactive', 'expired', 'invalid'
    expiry: Optional[datetime] = None
    last_check: datetime
    token: Optional[str] = None
