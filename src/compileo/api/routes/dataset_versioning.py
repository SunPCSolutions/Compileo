"""Dataset Versioning routes for the Compileo API."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import DatasetVersionRepository

# Create router
router = APIRouter()

# Pydantic models
class VersionListResponse(BaseModel):
    versions: list
    total: int

class VersionComparisonRequest(BaseModel):
    project_id: int
    dataset_name: str
    version1: str
    version2: str

class VersionIncrementRequest(BaseModel):
    project_id: int
    dataset_name: str
    version_type: str = "patch"
    description: Optional[str] = None

class RollbackRequest(BaseModel):
    project_id: int
    dataset_name: str
    target_version: str

# Dependency to get database connection
def get_db():
    return get_db_connection()

@router.get("/", response_model=VersionListResponse)
async def list_versions(
    project_id: int,
    dataset_name: str,
    active_only: bool = True,
    db=Depends(get_db)
):
    """List all versions of a dataset."""
    try:
        version_repo = DatasetVersionRepository(db)

        versions = version_repo.get_versions_by_project(project_id, dataset_name, active_only)

        return VersionListResponse(
            versions=[{
                "version": v[3],  # version string
                "total_entries": v[9],  # total_entries
                "is_active": v[13],  # is_active
                "created_at": v[12],  # created_at
                "description": v[6]  # description
            } for v in versions],
            total=len(versions)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list versions: {str(e)}")

@router.post("/compare")
async def compare_versions(request: VersionComparisonRequest, db=Depends(get_db)):
    """Compare two dataset versions."""
    try:
        version_repo = DatasetVersionRepository(db)

        comparison = version_repo.compare_versions(
            request.project_id,
            request.dataset_name,
            request.version1,
            request.version2
        )

        return comparison

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare versions: {str(e)}")

@router.post("/rollback")
async def rollback_version(request: RollbackRequest, db=Depends(get_db)):
    """Rollback dataset to a previous version."""
    try:
        version_repo = DatasetVersionRepository(db)

        success = version_repo.rollback_to_version(
            request.project_id,
            request.dataset_name,
            request.target_version
        )

        if success:
            return {"message": f"Successfully rolled back to version {request.target_version}"}
        else:
            raise HTTPException(status_code=400, detail="Rollback failed")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rollback version: {str(e)}")

@router.post("/increment")
async def increment_version(request: VersionIncrementRequest, db=Depends(get_db)):
    """Increment the version of a dataset."""
    try:
        version_repo = DatasetVersionRepository(db)

        new_version = version_repo.increment_version(
            request.project_id,
            request.dataset_name,
            request.version_type
        )

        # Create new version record
        version_id = version_repo.create_version(
            project_id=request.project_id,
            version=new_version,
            major_version=int(new_version.split('.')[0]),
            minor_version=int(new_version.split('.')[1]),
            patch_version=int(new_version.split('.')[2]),
            dataset_name=request.dataset_name,
            description=request.description
        )

        return {
            "new_version": new_version,
            "version_id": version_id,
            "message": f"Version incremented to {new_version}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to increment version: {str(e)}")

@router.get("/latest")
async def get_latest_version(
    project_id: int,
    dataset_name: str,
    db=Depends(get_db)
):
    """Get the latest version of a dataset."""
    try:
        version_repo = DatasetVersionRepository(db)

        latest_version = version_repo.get_latest_version(project_id, dataset_name)

        if not latest_version:
            raise HTTPException(status_code=404, detail=f"No versions found for dataset '{dataset_name}'")

        return {"latest_version": latest_version}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get latest version: {str(e)}")