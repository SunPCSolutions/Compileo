"""Prompts routes for the Compileo API."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import PromptRepository

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
    id: int
    name: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class PromptListResponse(BaseModel):
    prompts: List[PromptResponse]
    total: int

# Dependency to get database connection
def get_db():
    return get_db_connection()

@router.get("/", response_model=PromptListResponse)
async def list_prompts(
    name_filter: Optional[str] = None,
    limit: int = 50,
    db=Depends(get_db)
):
    """List all prompts with optional filtering."""
    try:
        prompt_repo = PromptRepository(db)

        # TODO: Implement proper listing with filtering
        # For now, return empty list
        return PromptListResponse(prompts=[], total=0)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list prompts: {str(e)}")

@router.post("/", response_model=PromptResponse)
async def create_prompt(prompt: PromptCreate, db=Depends(get_db)):
    """Create a new prompt."""
    try:
        prompt_repo = PromptRepository(db)

        # Check if prompt name already exists
        existing = prompt_repo.get_by_name(prompt.name)
        if existing:
            raise HTTPException(status_code=400, detail=f"Prompt with name '{prompt.name}' already exists")

        prompt_id = prompt_repo.create(prompt.name, prompt.content)

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
async def get_prompt(prompt_id: int, db=Depends(get_db)):
    """Get a prompt by ID."""
    try:
        prompt_repo = PromptRepository(db)

        # TODO: Implement get by ID method in repository
        # For now, return mock data
        return PromptResponse(
            id=prompt_id,
            name=f"Prompt {prompt_id}",
            content="Sample prompt content",
            created_at=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prompt: {str(e)}")

@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: int, prompt_update: PromptUpdate, db=Depends(get_db)):
    """Update a prompt."""
    try:
        prompt_repo = PromptRepository(db)

        # TODO: Implement update logic
        return await get_prompt(prompt_id, db)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update prompt: {str(e)}")

@router.delete("/{prompt_id}")
async def delete_prompt(prompt_id: int, db=Depends(get_db)):
    """Delete a prompt."""
    try:
        prompt_repo = PromptRepository(db)

        prompt_repo.delete(prompt_id)

        return {"message": f"Prompt {prompt_id} deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete prompt: {str(e)}")

@router.get("/by-name/{name}", response_model=PromptResponse)
async def get_prompt_by_name(name: str, db=Depends(get_db)):
    """Get a prompt by name."""
    try:
        prompt_repo = PromptRepository(db)

        prompt_data = prompt_repo.get_by_name(name)
        if not prompt_data:
            raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")

        return PromptResponse(
            id=prompt_data[0],
            name=prompt_data[1],
            content=prompt_data[2],
            created_at=datetime.utcnow()  # TODO: Get actual timestamp
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prompt by name: {str(e)}")