"""Projects routes for the Compileo API."""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from datetime import datetime

from ...storage.src.database import get_tenant_db_connection
from ...storage.src.project.database_repositories import ProjectRepository, DocumentRepository, DatasetJobRepository
from ..core.dependencies import get_current_active_user
from ..models.user import User as UserSchema

# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

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

@router.get("", response_model=ProjectListResponse)
def list_projects(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=1000),
    current_user: UserSchema = Depends(get_current_active_user)
  ):
    """List all projects for the current user with pagination."""
    try:
        project_repo = ProjectRepository(current_user.id)
        document_repo = DocumentRepository(current_user.id)
        dataset_repo = DatasetJobRepository(current_user.id)

        # Get all projects for current user
        all_projects = project_repo.get_all_projects()
        total = len(all_projects)

        # Apply pagination
        offset = (page - 1) * per_page
        paginated_projects = all_projects[offset:offset + per_page]

        projects = []
        for project in paginated_projects:
            try:
                project_id = project['id']
                name = project['name']
                created_at = project['created_at']

                # Count documents and datasets for this project
                document_count = len(document_repo.get_documents_by_project_id(project_id))
                
                # For datasets, count from dataset_versions via database
                with get_tenant_db_connection(current_user.id) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM dataset_versions WHERE project_id = ? AND owner_id = ?",
                        (project_id, current_user.id)
                    )
                    dataset_count = cursor.fetchone()[0]

                projects.append(ProjectResponse(
                    id=str(project_id),
                    name=name,
                    created_at=created_at,
                    document_count=document_count,
                    dataset_count=dataset_count
                ))
            except Exception as row_error:
                import logging
                logging.getLogger(__name__).error(f"Error processing project: {row_error}")
                raise

        return ProjectListResponse(
            projects=projects,
            total=total,
            page=page,
            per_page=per_page
        )

    except Exception as e:
        import traceback
        error_detail = f"Failed to list projects: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")

@router.post("", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Create a new project."""
    try:
        project_repo = ProjectRepository(current_user.id)

        # Check if project name already exists for current user
        existing_id = project_repo.get_project_by_name(project.name)
        if existing_id:
            raise HTTPException(status_code=400, detail=f"Project with name '{project.name}' already exists")

        # Create project with owner_id
        project_id = project_repo.create_project(project.name)

        return ProjectResponse(
            id=str(project_id),
            name=project.name,
            description=project.description,
            created_at=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"ERROR: Failed to create project: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get project details by ID for current user."""
    try:
        project_repo = ProjectRepository(current_user.id)
        document_repo = DocumentRepository(current_user.id)

        # Get project basic info
        project = project_repo.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        name = project['name']
        created_at = project['created_at']

        # Count documents and datasets
        document_count = len(document_repo.get_documents_by_project_id(project_id))
        
        with get_tenant_db_connection(current_user.id) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM dataset_versions WHERE project_id = ? AND owner_id = ?",
                (project_id, current_user.id)
            )
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
async def update_project(
    project_id: str,
    project_update: ProjectUpdate,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Update project details."""
    try:
        project_repo = ProjectRepository(current_user.id)

        # Check if project exists and belongs to user
        project = project_repo.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        current_name = project['name']

        # Update project name if provided
        if project_update.name and project_update.name != current_name:
            # Check if new name already exists for user
            existing_id = project_repo.get_project_by_name(project_update.name)
            if existing_id and existing_id != project_id:
                raise HTTPException(status_code=400, detail=f"Project with name '{project_update.name}' already exists")

            # Update name via raw SQL (repository doesn't have update method)
            with get_tenant_db_connection(current_user.id) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE projects SET name = ? WHERE id = ? AND owner_id = ?",
                    (project_update.name, project_id, current_user.id)
                )
                conn.commit()

        # Get updated project info
        return await get_project(project_id, current_user)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")

@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Delete a project and all associated data."""
    try:
        logger.info(f"Starting deletion of project {project_id} for user {current_user.id}")

        with get_tenant_db_connection(current_user.id) as conn:
            cursor = conn.cursor()

            # Check if project exists and belongs to user
            logger.debug(f"Checking if project {project_id} exists")
            cursor.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?", (project_id, current_user.id))
            row = cursor.fetchone()

            if not row:
                logger.warning(f"Project {project_id} not found or doesn't belong to user")
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

            logger.info(f"Found {len(document_rows)} document files, {len(taxonomy_rows)} taxonomies, {len(output_rows)} output files")

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
                    logger.warning(f"Warning: Error deleting taxonomy {taxonomy_id}: {e}")

            # 4. Check for embed records that reference chunks (diagnostic logging)
            logger.debug(f"Checking for embed records referencing chunks for project {project_id}")
            cursor.execute("SELECT COUNT(*) FROM embed WHERE document_id IN (SELECT id FROM documents WHERE project_id = ?)", (project_id,))
            embed_count = cursor.fetchone()[0]
            logger.debug(f"Found {embed_count} embed records referencing chunks for project {project_id}")
            
            if embed_count > 0:
                logger.warning(f"Deleting {embed_count} embed records before chunk deletion")
                cursor.execute("DELETE FROM embed WHERE document_id IN (SELECT id FROM documents WHERE project_id = ?)", (project_id,))

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
            conn.commit()
            logger.info(f"Project record deleted successfully")

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
                        logger.warning(f"Warning: Failed to remove document file {file_path}: {e}")

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
                        logger.warning(f"Warning: Failed to remove output file {file_path}: {e}")

            # Clean up project-specific directories (database-mediated - we know these should exist for the project)
            directories_to_clean = [
                f"storage/{current_user.id}/uploads/{project_id}",
                f"storage/{current_user.id}/parsed/{project_id}",
                f"storage/{current_user.id}/chunks/{project_id}",
                f"storage/{current_user.id}/taxonomy/{project_id}",
                f"storage/{current_user.id}/extract/{project_id}",
                f"storage/{current_user.id}/datasets/{project_id}",
                f"storage/{current_user.id}/dataqual/{project_id}"
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
                    logger.warning(f"Warning: Failed to remove project directory {full_dir_path}: {e}")

        return {"message": f"Project {project_id} and all associated data deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"DEBUG: Error deleting project: {e}")
        logger.debug(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

@router.delete("")
async def bulk_delete_projects(
    bulk_delete: ProjectBulkDelete,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Delete multiple projects and all their associated data."""
    if not bulk_delete.project_ids:
        raise HTTPException(status_code=400, detail="No project IDs provided")

    deleted_projects = []
    failed_projects = []

    for project_id in bulk_delete.project_ids:
        try:
            # Reuse the existing delete logic
            await delete_project(project_id, current_user)
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
async def get_project_documents(
    project_id: str,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get all documents for a project for current user."""
    try:
        document_repo = DocumentRepository(current_user.id)

        # Check if project exists and belongs to user
        project_repo = ProjectRepository(current_user.id)
        project = project_repo.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        documents = document_repo.get_documents_by_project_id(project_id)

        return {"documents": documents}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project documents: {str(e)}")

@router.get("/{project_id}/datasets")
async def get_project_datasets(
    project_id: str,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get all datasets for a project for current user."""
    try:
        # Check if project exists and belongs to user
        project_repo = ProjectRepository(current_user.id)
        project = project_repo.get_project_by_id(project_id)

        if not project:
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        # Get dataset versions for this project for current user
        with get_tenant_db_connection(current_user.id) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT dv.*, dp.purpose, dp.audience, dp.dataset_format
                FROM dataset_versions dv
                LEFT JOIN dataset_parameters dp ON dv.project_id = dp.project_id
                WHERE dv.project_id = ? AND dv.owner_id = ?
                ORDER BY dv.created_at DESC
            """, (project_id, current_user.id))

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
