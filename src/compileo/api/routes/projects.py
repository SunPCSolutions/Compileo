"""Projects routes for the Compileo API."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import ProjectRepository, DocumentRepository, DatasetJobRepository
from ...core.logging import get_logger

# Create router
router = APIRouter()
logger = get_logger(__name__)

# Pydantic models
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    document_count: int = 0
    dataset_count: int = 0
    status: str = "active"

class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]
    total: int
    page: int
    per_page: int

class ProjectBulkDelete(BaseModel):
    project_ids: List[str]

# Dependency to get database connection
def get_db():
    # Use the centralized database connection function
    from ...storage.src.database import get_db_connection
    return get_db_connection()

@router.get("", response_model=ProjectListResponse)
def list_projects(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=1000)
  ):
    """List all projects with pagination."""
    db = get_db()
    try:
        project_repo = ProjectRepository(db)
        document_repo = DocumentRepository(db)
        dataset_repo = DatasetJobRepository(db)

        # Get total count
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM projects")
        total = cursor.fetchone()[0]

        # Get projects with pagination
        offset = (page - 1) * per_page
        cursor.execute("""
            SELECT id, name, created_at
            FROM projects
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (per_page, offset))
        project_rows = cursor.fetchall()

        projects = []
        for row in project_rows:
            project_id, name, created_at = row

            # Count documents and datasets for this project
            document_count = len(document_repo.get_documents_by_project_id(project_id))
            # For datasets, count dataset_versions entries for this project
            cursor.execute("SELECT COUNT(*) FROM dataset_versions WHERE project_id = ?", (project_id,))
            dataset_count = cursor.fetchone()[0]

            projects.append(ProjectResponse(
                id=project_id,
                name=name,
                created_at=created_at,
                document_count=document_count,
                dataset_count=dataset_count
            ))

        return ProjectListResponse(
            projects=projects,
            total=total,
            page=page,
            per_page=per_page
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")
    finally:
        db.close()

@router.post("", response_model=ProjectResponse)
async def create_project(project: ProjectCreate, db=Depends(get_db)):
    """Create a new project."""
    try:
        project_repo = ProjectRepository(db)

        # Check if project name already exists
        existing_id = project_repo.get_project_by_name(project.name)
        if existing_id:
            raise HTTPException(status_code=400, detail=f"Project with name '{project.name}' already exists")

        # Create project
        project_id = project_repo.create_project(project.name)

        return ProjectResponse(
            id=project_id,
            name=project.name,
            description=project.description,
            created_at=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db=Depends(get_db)):
    """Get project details by ID."""
    try:
        project_repo = ProjectRepository(db)
        document_repo = DocumentRepository(db)

        # Get project basic info
        cursor = db.cursor()
        cursor.execute("SELECT name, created_at FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        name, created_at = row

        # Count documents and datasets
        document_count = len(document_repo.get_documents_by_project_id(project_id))
        # Count dataset_versions entries for this project
        cursor.execute("SELECT COUNT(*) FROM dataset_versions WHERE project_id = ?", (project_id,))
        dataset_count = cursor.fetchone()[0]

        return ProjectResponse(
            id=project_id,
            name=name,
            created_at=created_at,
            document_count=document_count,
            dataset_count=dataset_count
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, project_update: ProjectUpdate, db=Depends(get_db)):
    """Update project details."""
    try:
        project_repo = ProjectRepository(db)

        # Check if project exists
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        current_name = row[0]

        # Update project name if provided
        if project_update.name and project_update.name != current_name:
            # Check if new name already exists
            existing_id = project_repo.get_project_by_name(project_update.name)
            if existing_id and existing_id != project_id:
                raise HTTPException(status_code=400, detail=f"Project with name '{project_update.name}' already exists")

            # Update name in database
            cursor.execute("UPDATE projects SET name = ? WHERE id = ?", (project_update.name, project_id))
            db.commit()

        # Get updated project info
        return await get_project(project_id, db)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")

@router.delete("/{project_id}")
async def delete_project(project_id: str, db=Depends(get_db)):
    """Delete a project and all associated data."""
    try:
        logger.info(f"Starting deletion of project {project_id}")
        cursor = db.cursor()

        # Check if project exists
        logger.debug(f"Checking if project {project_id} exists")
        cursor.execute("SELECT name FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()

        if not row:
            logger.debug(f"Project {project_id} not found")
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        project_name = row[0]
        logger.debug(f"Found project {project_id}: {project_name}")

        # Get all file paths BEFORE deleting database records
        logger.debug(f"Querying file paths for project {project_id}")

        # Get document files
        cursor.execute("SELECT source_file_path FROM documents WHERE project_id = ?", (project_id,))
        document_rows = cursor.fetchall()

        # Get parsed files
        cursor.execute("""
            SELECT pf.file_path
            FROM parsed_files pf
            JOIN parsed_documents pd ON pf.parsed_document_id = pd.id
            JOIN documents d ON pd.document_id = d.id
            WHERE d.project_id = ?
        """, (project_id,))
        parsed_file_rows = cursor.fetchall()

        # Get chunk files
        cursor.execute("""
            SELECT c.file_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.project_id = ?
        """, (project_id,))
        chunk_rows = cursor.fetchall()

        # Get taxonomy records for this project
        cursor.execute("SELECT id, file_path FROM taxonomies WHERE project_id = ?", (project_id,))
        taxonomy_rows = cursor.fetchall()

        # Get extraction result files
        cursor.execute("SELECT file_path FROM extraction_results WHERE project_id = ?", (project_id,))
        extraction_rows = cursor.fetchall()

        # Get dataset files
        cursor.execute("SELECT file_path FROM dataset_versions WHERE project_id = ?", (project_id,))
        dataset_rows = cursor.fetchall()

        # Combine all file paths
        output_rows = parsed_file_rows + chunk_rows + extraction_rows + dataset_rows

        logger.debug(f"Found {len(document_rows)} document files, {len(taxonomy_rows)} taxonomies, {len(output_rows)} output files")

        # Delete in reverse pipeline order to maintain referential integrity

        # 1. Delete dataset-related records
        logger.debug(f"Deleting dataset records for project {project_id}")
        cursor.execute("DELETE FROM dataset_lineage WHERE dataset_version_id IN (SELECT id FROM dataset_versions WHERE project_id = ?)", (project_id,))
        cursor.execute("DELETE FROM dataset_changes WHERE dataset_version_id IN (SELECT id FROM dataset_versions WHERE project_id = ?)", (project_id,))
        cursor.execute("DELETE FROM dataset_versions WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM dataset_parameters WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM dataset_jobs WHERE project_id = ?", (project_id,))

        # 2. Delete extraction records
        logger.debug(f"Deleting extraction records for project {project_id}")
        cursor.execute("DELETE FROM extraction_results WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM extraction_jobs WHERE project_id = ?", (project_id,))

        # 3. Delete taxonomy records and files
        logger.debug(f"Deleting {len(taxonomy_rows)} taxonomy records for project {project_id}")
        for taxonomy_row in taxonomy_rows:
            taxonomy_id, taxonomy_file_path = taxonomy_row
            try:
                # Delete taxonomy file from filesystem
                if taxonomy_file_path:
                    import os
                    from pathlib import Path
                    taxonomy_file_path_obj = Path(taxonomy_file_path)
                    if taxonomy_file_path_obj.exists():
                        taxonomy_file_path_obj.unlink()
                        logger.debug(f"Successfully deleted taxonomy file: {taxonomy_file_path}")
                    else:
                        logger.debug(f"Taxonomy file not found: {taxonomy_file_path}")

                # Delete taxonomy record from database
                cursor.execute("DELETE FROM taxonomies WHERE id = ?", (taxonomy_id,))
                logger.debug(f"Successfully deleted taxonomy record {taxonomy_id}")
            except Exception as e:
                logger.warning(f"Error deleting taxonomy {taxonomy_id}: {e}")

        # 4. Delete chunks
        logger.debug(f"Deleting chunk records for project {project_id}")
        cursor.execute("DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE project_id = ?)", (project_id,))

        # 5. Delete parsed files and documents
        logger.debug(f"Deleting parsed records for project {project_id}")
        cursor.execute("DELETE FROM parsed_files WHERE parsed_document_id IN (SELECT id FROM parsed_documents WHERE project_id = ?)", (project_id,))
        cursor.execute("DELETE FROM parsed_documents WHERE project_id = ?", (project_id,))

        # 6. Delete prompts
        cursor.execute("DELETE FROM prompts WHERE project_id = ?", (project_id,))

        # 7. Delete processed outputs
        cursor.execute("DELETE FROM processed_outputs WHERE project_id = ?", (project_id,))

        # 8. Delete documents (this will cascade to related records)
        cursor.execute("DELETE FROM documents WHERE project_id = ?", (project_id,))

        # 9. Finally delete the project
        logger.debug(f"Deleting project record {project_id}")
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        db.commit()
        logger.debug(f"Project record deleted successfully")

        # Clean up filesystem files
        logger.debug(f"Starting filesystem cleanup for project {project_id}")
        import os
        import shutil

        # Get base directory for absolute path calculations
        base_dir = os.getcwd()
        logger.debug(f"Base directory: {base_dir}")

        # Clean up document files
        for doc_row in document_rows:
            file_path = doc_row[0]
            if file_path:
                # Convert relative paths to absolute paths
                if not os.path.isabs(file_path):
                    file_path = os.path.join(base_dir, file_path)

                try:
                    # Remove the file
                    os.remove(file_path)
                    logger.debug(f"Successfully removed document file: {file_path}")
                    # Try to remove the parent directory if it's a project-specific directory
                    parent_dir = os.path.dirname(file_path)
                    if parent_dir != os.path.join(base_dir, "uploads") and os.path.exists(parent_dir):
                        # Check if directory is empty
                        if not os.listdir(parent_dir):
                            os.rmdir(parent_dir)
                            logger.debug(f"Successfully removed empty directory: {parent_dir}")
                except FileNotFoundError:
                    logger.debug(f"Document file already removed: {file_path}")
                except Exception as e:
                    # Log but don't fail the operation
                    logger.warning(f"Failed to remove document file {file_path}: {e}")

        # Clean up processed output files
        for output_row in output_rows:
            file_path = output_row[0]
            if file_path:
                # Convert relative paths to absolute paths
                if not os.path.isabs(file_path):
                    file_path = os.path.join(base_dir, file_path)

                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.debug(f"Successfully removed output file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.debug(f"Successfully removed output directory: {file_path}")
                except FileNotFoundError:
                    logger.debug(f"Output file already removed: {file_path}")
                except Exception as e:
                    # Log but don't fail the operation
                    logger.warning(f"Failed to remove output file {file_path}: {e}")

        # Clean up project-specific directories (database-mediated - we know these should exist for the project)
        directories_to_clean = [
            f"storage/uploads/{project_id}",
            f"storage/parsed/{project_id}",
            f"storage/chunks/{project_id}",
            f"storage/taxonomy/{project_id}",
            f"storage/extract/{project_id}",
            f"storage/datasets/{project_id}",
            f"storage/dataqual/{project_id}"
        ]

        for dir_path in directories_to_clean:
            full_dir_path = os.path.join(base_dir, dir_path)
            try:
                # Attempt to remove directory - this is mediated by project structure knowledge
                # We know these directories should exist for projects, so we attempt cleanup
                shutil.rmtree(full_dir_path)
                logger.debug(f"Successfully removed project directory: {full_dir_path}")
            except FileNotFoundError:
                # Directory doesn't exist - this is fine, nothing to clean up
                logger.debug(f"Directory does not exist (already clean): {full_dir_path}")
            except Exception as e:
                # Other errors (permissions, etc.) - log but don't fail
                logger.warning(f"Failed to remove project directory {full_dir_path}: {e}")

        return {"message": f"Project {project_id} and all associated data deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

@router.delete("")
async def bulk_delete_projects(bulk_delete: ProjectBulkDelete, db=Depends(get_db)):
    """Delete multiple projects and all their associated data."""
    if not bulk_delete.project_ids:
        raise HTTPException(status_code=400, detail="No project IDs provided")

    deleted_projects = []
    failed_projects = []

    for project_id in bulk_delete.project_ids:
        try:
            # Reuse the existing delete logic
            await delete_project(project_id, db)
            deleted_projects.append(project_id)
        except HTTPException as e:
            failed_projects.append({"id": project_id, "error": e.detail})
        except Exception as e:
            failed_projects.append({"id": project_id, "error": str(e)})

    if failed_projects:
        # Some deletions failed, return partial success
        return {
            "message": f"Deleted {len(deleted_projects)} projects successfully, {len(failed_projects)} failed",
            "deleted": deleted_projects,
            "failed": failed_projects
        }
    else:
        # All deletions successful
        return {
            "message": f"Successfully deleted {len(deleted_projects)} projects and all associated data",
            "deleted": deleted_projects
        }

@router.get("/{project_id}/documents")
async def get_project_documents(project_id: str, db=Depends(get_db)):
    """Get all documents for a project."""
    try:
        document_repo = DocumentRepository(db)

        # Check if project exists
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ?", (project_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        documents = document_repo.get_documents_by_project_id(project_id)

        return {"documents": documents}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project documents: {str(e)}")

@router.get("/{project_id}/datasets")
async def get_project_datasets(project_id: str, db=Depends(get_db)):
    """Get all datasets for a project."""
    try:
        # Check if project exists
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        # Get dataset versions for this project
        cursor.execute("""
            SELECT dv.*, dp.purpose, dp.audience, dp.dataset_format
            FROM dataset_versions dv
            LEFT JOIN dataset_parameters dp ON dv.project_id = dp.project_id
            WHERE dv.project_id = ?
            ORDER BY dv.created_at DESC
        """, (project_id,))

        datasets = []
        for row in cursor.fetchall():
            datasets.append({
                'id': row['id'],
                'version': row['version'],
                'dataset_name': row['dataset_name'],
                'total_entries': row['total_entries'],
                'file_path': row['file_path'],
                'created_at': row['created_at'],
                'purpose': row.get('purpose'),
                'audience': row.get('audience'),
                'format': row.get('dataset_format')
            })

        return {"datasets": datasets}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project datasets: {str(e)}")
