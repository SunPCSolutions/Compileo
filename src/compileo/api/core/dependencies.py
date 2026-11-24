from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from ..core.auth import SECRET_KEY, ALGORITHM, TokenData
from ..services.user_service import UserService
from ..models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
        
    user = UserService.get_user_by_id(user_id=token_data.user_id)
    if user is None:
        raise credentials_exception
    return User.model_validate(user)

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="The user doesn't have enough privileges")
    return current_user

async def require_license(request: Request, current_user: User = Depends(get_current_active_user)):
    """Dependency to check if the current user has a valid license."""
    from ...features.licensing.manager import LicenseManager
    license_manager = LicenseManager()
    status = license_manager.get_local_status(owner_id=current_user.id)
    
    if not status or status.status != 'active':
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="A valid license is required to perform this operation"
        )
