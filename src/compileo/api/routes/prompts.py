"""Prompts routes for the Compileo API."""

import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import PromptRepository
from ..core.database import get_db
from ..core.dependencies import get_current_active_user
from ..models.user import User as UserSchema

# Create router
router = APIRouter()

# Pydantic models
class PromptCreate(BaseModel):
    name: str
    content: str

class PromptUpdate(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None

class PromptResponse(BaseModel):
    id: str
    name: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class PromptListResponse(BaseModel):
    prompts: List[PromptResponse]
    total: int

@router.get("/", response_model=PromptListResponse)
async def list_prompts(
    project_id: str,
    name_filter: Optional[str] = None,
    limit: int = 50,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """List all prompts for a project and current user with optional filtering."""
    try:
        prompt_repo = PromptRepository(current_user.id)
        prompts = prompt_repo.get_prompts_by_project(project_id)
        
        if name_filter:
            prompts = [p for p in prompts if name_filter.lower() in p['name'].lower()]
            
        return PromptListResponse(
            prompts=prompts[:limit],
            total=len(prompts)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list prompts: {str(e)}")

@router.post("/", response_model=PromptResponse)
async def create_prompt(
    prompt: PromptCreate, 
    project_id: str,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Create a new prompt for a project and current user."""
    try:
        prompt_repo = PromptRepository(current_user.id)

        # Create prompt
        prompt_id = prompt_repo.create({
            "id": str(uuid.uuid4()),
            "owner_id": current_user.id,
            "project_id": project_id,
            "name": prompt.name,
            "content": prompt.content,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })

        return PromptResponse(
            id=prompt_id,
            name=prompt.name,
            content=prompt.content,
            created_at=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create prompt: {str(e)}")

@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(
    prompt_id: str, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get a prompt by ID for current user."""
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM prompts WHERE id = ? AND owner_id = ?", (prompt_id, current_user.id))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Prompt with ID {prompt_id} not found")

        return PromptResponse(**dict(row))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prompt: {str(e)}")

@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(
    prompt_id: str, 
    prompt_update: PromptUpdate, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Update a prompt for current user."""
    try:
        cursor = db.cursor()
        # Check ownership
        cursor.execute("SELECT name FROM prompts WHERE id = ? AND owner_id = ?", (prompt_id, current_user.id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Prompt with ID {prompt_id} not found")

        update_fields = []
        params = []
        if prompt_update.name:
            update_fields.append("name = ?")
            params.append(prompt_update.name)
        if prompt_update.content:
            update_fields.append("content = ?")
            params.append(prompt_update.content)
            
        if update_fields:
            update_fields.append("updated_at = ?")
            params.append(datetime.utcnow())
            query = f"UPDATE prompts SET {', '.join(update_fields)} WHERE id = ? AND owner_id = ?"
            params.extend([prompt_id, current_user.id])
            cursor.execute(query, params)
            db.commit()

        return await get_prompt(prompt_id, db, current_user)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update prompt: {str(e)}")

@router.delete("/{prompt_id}")
async def delete_prompt(
    prompt_id: str, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Delete a prompt for current user."""
    try:
        cursor = db.cursor()
        cursor.execute("DELETE FROM prompts WHERE id = ? AND owner_id = ?", (prompt_id, current_user.id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Prompt with ID {prompt_id} not found")
        db.commit()
        return {"message": f"Prompt {prompt_id} deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete prompt: {str(e)}")

@router.get("/by-name/{name}", response_model=PromptResponse)
async def get_prompt_by_name(
    name: str,
    project_id: str,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get a prompt by name for current user (or system default)."""
    try:
        prompt_repo = PromptRepository(current_user.id)
        prompt = prompt_repo.get_by_name(name, project_id=project_id)
        
        if not prompt:
            raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")

        return PromptResponse(**prompt)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prompt by name: {str(e)}")