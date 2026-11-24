from typing import Optional
from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    email: EmailStr
    username: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: Optional[str] = None

class UserInDB(UserBase):
    id: str
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False

class User(UserBase):
    id: str
    is_active: bool
    is_admin: bool

    class Config:
        from_attributes = True

class UserQuota(BaseModel):
    user_id: str
    max_projects: int
    max_documents_per_project: int
    max_storage_mb: int
    max_concurrent_jobs: int
