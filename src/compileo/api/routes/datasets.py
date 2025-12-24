"""Datasets routes for the Compileo API."""

import os
import uuid
import json
from typing import List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, validator
from datetime import datetime
import io
import zipfile
from pathlib import Path

from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import ProjectRepository, DatasetParameterRepository, PromptRepository, ChunkRepository
from ...storage.src.project.file_manager import FileManager
from ...core.logging import get_logger

# Create router
router = APIRouter()
logger = get_logger(__name__)

# Pydantic models
class DatasetGenerateRequest(BaseModel):
    project_id: str
    prompt_name: str = "default"
    custom_prompt: Optional[str] = None  # Custom prompt content from GUI
    generation_mode: str = "default"  # Mode selected in GUI (default, question, answer, summarization)
    format_type: str = "jsonl"  # "jsonl" or "parquet"
    concurrency: int = 1  # Number of parallel workers per batch
    batch_size: int = 50  # Number of chunks to process per batch (0 = all at once)
    include_evaluation_sets: bool = False
    taxonomy_project: Optional[str] = None
    taxonomy_name: Optional[str] = None
    data_source: str = "Chunks Only"  # Data source: "Chunks Only", "Taxonomy", "Extract"
    extraction_file_id: Optional[str] = None  # Specific extraction file ID when data_source is "Extract"
    selected_categories: Optional[List[str]] = None  # Selected category IDs for filtering when data_source is "Extract"
    only_validated: bool = False  # Filter extraction results to only include validated data
    output_dir: str = "."
    analyze_quality: bool = True
    quality_threshold: float = 0.7
    enable_versioning: bool = False
    dataset_name: Optional[str] = None
    run_benchmarks: bool = False
    benchmark_suite: str = "glue"
    parsing_model: str = "gemini"  # AI model for document parsing
    chunking_model: str = "gemini"  # AI model for text chunking
    classification_model: str = "gemini"  # AI model for document classification
    datasets_per_chunk: int = 3  # Maximum number of datasets to generate per text chunk
    purpose: Optional[str] = None
    audience: Optional[str] = None
    extraction_rules: Optional[str] = None
    dataset_format: Optional[str] = None
    question_style: Optional[str] = None
    answer_style: Optional[str] = None
    negativity_ratio: Optional[float] = None
    data_augmentation: Optional[str] = None
    custom_audience: Optional[str] = None
    custom_purpose: Optional[str] = None
    complexity_level: Optional[str] = None
    domain: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: int  # 0-100
    current_step: str
    estimated_completion: Optional[datetime] = None
    result: Optional[dict] = None
    error: Optional[str] = None

class DatasetEntry(BaseModel):
    id: str
    question: str
    answer: str
    category: Optional[str] = None
    quality_score: Optional[float] = None
    difficulty: Optional[str] = None
    source_chunk: Optional[str] = None
    metadata: Optional[dict] = None

class DatasetResponse(BaseModel):
    id: str
    name: str
    entries: List[DatasetEntry]
    total_entries: int
    created_at: datetime
    format_type: str
    quality_summary: Optional[dict] = None
class DatasetParameterRequest(BaseModel):
    project_id: str
    purpose: str
    audience: str
    extraction_rules: str = "default"
    dataset_format: str
    question_style: str = "factual"
    answer_style: str = "comprehensive"
    negativity_ratio: float = 0.1
    data_augmentation: str = "none"
    # High-level prompt modifiers
    custom_audience: str = ""
    custom_purpose: str = ""
    complexity_level: str = "intermediate"
    domain: str = "general"

class DefaultPromptsResponse(BaseModel):
    """Default prompts for each generation mode."""
    prompts: dict

class HighLevelPromptsResponse(BaseModel):
    """High-level prompt configuration defaults and options."""
    audience_defaults: List[str]
    purpose_defaults: List[str]
    complexity_options: List[str]
    domain_defaults: List[str]

# Dependency to get database connection
def get_db():
    return get_db_connection()

# In-memory job storage (replace with proper job queue in production)
job_storage = {}

# Dependency to get repositories
def get_dataset_repositories(db=Depends(get_db)):
    from ...storage.src.project.database_repositories import DatasetJobRepository
    return {
        'project': ProjectRepository(db),
        'dataset_param': DatasetParameterRepository(db),
        'prompt': PromptRepository(db),
        'chunk': ChunkRepository(db),
        'dataset_job': DatasetJobRepository(db)
    }

@router.post("/generate", response_model=JobResponse)
async def generate_dataset(
    request: DatasetGenerateRequest,
    http_request: Request,
    db=Depends(get_db),
    repos=Depends(get_dataset_repositories)
):
    """Generate a dataset using the enhanced job queue."""
    from ...core.settings import backend_settings
    from ...features.jobhandle.models import JobType
    try:
        # Get job queue manager
        job_queue_manager = http_request.app.state.job_queue_manager
        if job_queue_manager is None:
            raise HTTPException(status_code=503, detail="Job queue manager not initialized")

        # Validate project exists
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ?", (request.project_id,))
        project_row = cursor.fetchone()
        if not project_row:
            raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")

        # Resolve models
        classification_model = (
            backend_settings.get_classification_gemini_model() if request.classification_model == "gemini" else
            backend_settings.get_classification_grok_model() if request.classification_model == "grok" else
            backend_settings.get_classification_openai_model() if request.classification_model == "openai" else
            backend_settings.get_classification_ollama_model() if request.classification_model == "ollama" else
            request.classification_model
        )
        generation_model = (
            backend_settings.get_generation_gemini_model() if request.classification_model == "gemini" else
            backend_settings.get_generation_grok_model() if request.classification_model == "grok" else
            backend_settings.get_generation_openai_model() if request.classification_model == "openai" else
            backend_settings.get_generation_ollama_model() if request.classification_model == "ollama" else
            None
        )

        # Save parameters
        param_request = DatasetParameterRequest(
            project_id=request.project_id,
            purpose=request.purpose or "",
            audience=request.audience or "",
            extraction_rules=request.extraction_rules or "default",
            dataset_format=request.dataset_format or "jsonl",
            question_style=request.question_style or "factual",
            answer_style=request.answer_style or "comprehensive",
            negativity_ratio=request.negativity_ratio or 0.1,
            data_augmentation=request.data_augmentation or "none",
            custom_audience=request.custom_audience or "",
            custom_purpose=request.custom_purpose or "",
            complexity_level=request.complexity_level or "intermediate",
            domain=request.domain or "general"
        )
        await save_dataset_parameters(param_request, db)

        # Create job parameters for the worker
        job_params = {
            'project_id': request.project_id,
            'generation_mode': request.generation_mode,
            'format_type': request.format_type,
            'data_source': request.data_source,
            'datasets_per_chunk': request.datasets_per_chunk,
            'concurrency': request.concurrency,
            'batch_size': request.batch_size,
            'custom_prompt': request.custom_prompt,
            'prompt_name': request.prompt_name,
            'taxonomy_project': request.taxonomy_project,
            'taxonomy_name': request.taxonomy_name,
            'extraction_file_id': request.extraction_file_id,
            'selected_categories': request.selected_categories,
            'only_validated': request.only_validated,
            'enable_versioning': request.enable_versioning,
            'dataset_name': request.dataset_name,
            'classification_provider': request.classification_model,
            'classification_model': classification_model,
            'generation_model': generation_model,
            'custom_audience': request.custom_audience,
            'custom_purpose': request.custom_purpose,
            'complexity_level': request.complexity_level,
            'domain': request.domain
        }

        # Submit to enhanced queue
        job_id = job_queue_manager.submit_job(
            job_type=JobType.DATASET_GENERATION,
            parameters=job_params
        )

        # Create job record in database
        job_data = {
            'id': job_id,
            'project_id': request.project_id,
            'job_type': 'dataset_generation',
            'status': 'pending',
            'parameters': job_params,
            'progress': 0.0,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        repos['dataset_job'].create_job(job_data)

        return JobResponse(
            job_id=job_id,
            message="Dataset generation job submitted to queue"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start dataset generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start dataset generation: {str(e)}")

@router.post("/generate-evaluation", response_model=JobResponse)
async def generate_evaluation_dataset(
    request: DatasetGenerateRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db)
):
    """Generate comprehensive evaluation datasets."""
    try:
        # Similar validation as generate_dataset
        project_repo = ProjectRepository(db)
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ?", (request.project_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")

        job_id = str(uuid.uuid4())
        job_storage[job_id] = JobStatus(
            job_id=job_id,
            status="pending",
            progress=0,
            current_step="Initializing evaluation dataset generation"
        )

        # Start background task
        background_tasks.add_task(
            process_evaluation_dataset_generation,
            job_id,
            request,
            db
        )

        return JobResponse(
            job_id=job_id,
            message="Evaluation dataset generation started"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation dataset generation: {str(e)}")

@router.get("/generate/{job_id}/status", response_model=JobStatus)
async def get_generation_status(job_id: str, repos=Depends(get_dataset_repositories)):
    """Get the status of a dataset generation job."""
    # Try in-memory storage first (active jobs in current session)
    if job_id in job_storage:
        return job_storage[job_id]
    
    # Fallback to database for persistent status tracking (cross-session/restart)
    try:
        job_repo = repos['dataset_job']
        job_record = job_repo.get_job_by_id(job_id)
        
        if job_record:
            # Map database record to JobStatus model
            return JobStatus(
                job_id=job_record['id'],
                status=job_record['status'],
                progress=int(job_record['progress'] * 100) if job_record['progress'] is not None else 0,
                current_step=job_record['status'].capitalize(),
                result=json.loads(job_record['result']) if job_record.get('result') else None,
                error=job_record.get('error')
            )
    except Exception as e:
        logger.debug(f"Error retrieving job from database: {e}")

    # Not found in memory or database
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, db=Depends(get_db)):
    """Get dataset details and entries."""
    try:
        # TODO: Implement dataset retrieval from storage
        # For now, return mock data
        return DatasetResponse(
            id=dataset_id,
            name=f"Dataset {dataset_id}",
            entries=[],
            total_entries=0,
            created_at=datetime.utcnow(),
            format_type="jsonl"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {str(e)}")

@router.get("/{dataset_id}/entries")
async def get_dataset_entries(
    dataset_id: str,
    page: int = 1,
    per_page: int = 50,
    filter_quality: Optional[float] = None,
    sort_by: str = "quality",
    db=Depends(get_db)
):
    """Get dataset entries with pagination and filtering."""
    try:
        # TODO: Implement dataset entries retrieval
        # For now, return mock data
        return {
            "entries": [],
            "total": 0,
            "page": page,
            "per_page": per_page,
            "quality_summary": {
                "average_score": 0.0,
                "distribution": {"high": 0, "medium": 0, "low": 0}
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset entries: {str(e)}")

@router.put("/{dataset_id}/entries/{entry_id}")
async def update_dataset_entry(
    dataset_id: str,
    entry_id: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    category: Optional[str] = None,
    feedback: Optional[str] = None,
    db=Depends(get_db)
):
    """Update a dataset entry (Enhancement 6 - Interactive Refinement)."""
    try:
        # TODO: Implement entry update logic
        return {"message": f"Entry {entry_id} updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update entry: {str(e)}")

@router.post("/{dataset_id}/feedback")
async def submit_feedback(
    dataset_id: str,
    entry_ids: List[str],
    feedback_type: str = "bulk_edit",
    comments: Optional[str] = None,
    rating: Optional[int] = None,
    db=Depends(get_db)
):
    """Submit user feedback for dataset entries."""
    try:
        # TODO: Implement feedback submission
        return {"message": f"Feedback submitted for {len(entry_ids)} entries"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.post("/{dataset_id}/regenerate")
async def regenerate_entries(
    dataset_id: str,
    entry_ids: List[str],
    regeneration_config: dict,
    db=Depends(get_db)
):
    """Regenerate specific dataset entries."""
    try:
        # TODO: Implement regeneration logic
        job_id = str(uuid.uuid4())
        return {
            "job_id": job_id,
            "message": f"Regeneration started for {len(entry_ids)} entries"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start regeneration: {str(e)}")


# In-memory storage for batch processing settings (replace with database in production)
_batch_processing_settings = {
    "batch_size": 50,
    "concurrency": 2
}


# Pydantic models for batch processing settings
class BatchProcessingSettings(BaseModel):
    batch_size: int = 50
    concurrency: int = 2

    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 0:
            raise ValueError('batch_size must be >= 0')
        return v

    @validator('concurrency')
    def validate_concurrency(cls, v):
        if v < 1 or v > 10:
            raise ValueError('concurrency must be between 1 and 10')
        return v


@router.post("/settings/batch-processing")
async def update_batch_processing_settings(
    settings: BatchProcessingSettings,
    db=Depends(get_db)
):
    """Update batch processing settings for dataset generation."""
    try:
        # Update settings in memory
        _batch_processing_settings["batch_size"] = settings.batch_size
        _batch_processing_settings["concurrency"] = settings.concurrency

        return {
            "message": "Batch processing settings updated successfully",
            "batch_size": settings.batch_size,
            "concurrency": settings.concurrency
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update batch processing settings: {str(e)}")


@router.get("/settings/batch-processing")
async def get_batch_processing_settings(db=Depends(get_db)):
    """Get current batch processing settings."""
    try:
        return {
            "batch_size": _batch_processing_settings["batch_size"],
            "concurrency": _batch_processing_settings["concurrency"],
            "description": "Current batch processing settings"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch processing settings: {str(e)}")

@router.get("/extraction-files/{project_id}")
async def get_extraction_files(project_id: str, db=Depends(get_db)):
    """Get available extraction files for a project."""
    try:
        logger.debug(f"Getting extraction files for project {project_id}")

        # Check if database connection is working
        if not db:
            logger.debug("No database connection")
            raise HTTPException(status_code=500, detail="Database connection not available")

        # Get extraction jobs for the project
        cursor = db.cursor()

        # First, let's get the jobs without the complex join to see if that works
        logger.debug(f"Executing query for project {project_id}")
        
        # Determine if project_id is int or string in database
        # If it's a string UUID in API but int in DB, we might need conversion or vice versa
        # Try both formats to be safe
        cursor.execute("""
            SELECT ej.id, ej.status, ej.created_at, ej.parameters
            FROM extraction_jobs ej
            WHERE (ej.project_id = ? OR ej.project_id = ?) AND ej.status = 'completed'
            ORDER BY ej.created_at DESC
        """, (project_id, str(project_id)))

        rows = cursor.fetchall()
        
        if not rows:
            logger.debug(f"No completed extraction jobs found for project {project_id}")
            # If no rows found, let's check if there are ANY jobs for this project, regardless of status
            cursor.execute("""
                SELECT ej.id, ej.status
                FROM extraction_jobs ej
                WHERE ej.project_id = ? OR ej.project_id = ?
            """, (project_id, str(project_id)))
            all_jobs = cursor.fetchall()
            logger.debug(f"Total jobs for project: {len(all_jobs)}")
            for job in all_jobs:
                logger.debug(f"Found job {job[0]} with status {job[1]}")
        
        if not rows:
            logger.debug(f"No extraction jobs found for project {project_id}")
            # If no rows found, let's check if there are ANY jobs for this project, regardless of status
            cursor.execute("""
                SELECT ej.id, ej.status
                FROM extraction_jobs ej
                WHERE ej.project_id = ? OR ej.project_id = ?
            """, (project_id, str(project_id)))
            all_jobs = cursor.fetchall()
            logger.debug(f"Total jobs for project: {len(all_jobs)}")
            for job in all_jobs:
                logger.debug(f"Found job {job[0]} with status {job[1]}")
        logger.debug(f"Found {len(rows)} completed extraction jobs")

        extraction_files = []
        for row in rows:
            job_id = row[0]  # The id column contains the job_id
            logger.debug(f"Processing job {job_id}")

            # Parse the parameters JSON to get extraction_type
            try:
                parameters = json.loads(row[3]) if row[3] else {}
                extraction_type = parameters.get('extraction_type', 'unknown')
                logger.debug(f"Parsed extraction_type: {extraction_type}")
            except json.JSONDecodeError as je:
                logger.debug(f"JSON decode error for job {job_id}: {je}")
                extraction_type = 'unknown'

            # Get entity count for this job
            try:
                cursor2 = db.cursor()
                cursor2.execute("SELECT COUNT(*) FROM extraction_results WHERE job_id = ?", (job_id,))
                entity_count_result = cursor2.fetchone()
                entity_count = entity_count_result[0] if entity_count_result else 0
                logger.debug(f"Entity count for job {job_id}: {entity_count}")
            except Exception as ce:
                logger.debug(f"Error counting entities for job {job_id}: {ce}")
                entity_count = 0

            extraction_files.append({
                "id": job_id,
                "job_id": job_id,  # Use the same value for both id and job_id
                "status": row[1],
                "created_at": row[2],
                "extraction_type": extraction_type,
                "entity_count": entity_count,
                "display_name": f"Job {job_id} - {extraction_type} ({entity_count} entities)"
            })

        logger.debug(f"Returning {len(extraction_files)} extraction files")
        return {"extraction_files": extraction_files}

    except Exception as e:
        logger.error(f"Failed to get extraction files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get extraction files: {str(e)}")


@router.get("/{dataset_id}/download")
async def download_dataset(dataset_id: str, db=Depends(get_db)):
    """
    Download dataset file(s).
    dataset_id can be a database ID (UUID or integer) or a filename string (for legacy support).
    """
    try:
        logger.debug(f"Attempting to download dataset: {dataset_id}")

        dataset_file = None
        media_type = "application/octet-stream"
        download_filename = dataset_id

        # 1. Try lookup by Database ID (UUID or Numeric) first
        is_filename = dataset_id.endswith(('.json', '.jsonl', '.parquet'))
        if dataset_id != "None" and not is_filename:
            try:
                cursor = db.cursor()
                # Try lookup directly by ID string (handles both numeric and UUID since column is TEXT)
                cursor.execute("SELECT file_path, dataset_name, version, metadata FROM dataset_versions WHERE id = ?", (dataset_id,))
                row = cursor.fetchone()
                logger.debug(f"DB lookup result for ID {dataset_id}: {row}")
                
                if row:
                    file_paths_raw = row[0]
                    name = row[1]
                    version = row[2]
                    metadata = json.loads(row[3]) if row[3] else {}
                    
                    # Determine format from metadata or filename
                    gen_params = metadata.get('generation_params', {})
                    # Try dataset_format first, then format_type, then extract from file path
                    format_type = gen_params.get('dataset_format') or gen_params.get('format_type')
                    
                    try:
                        file_paths = json.loads(file_paths_raw) if file_paths_raw else []
                        if isinstance(file_paths, str): # Handle case where it was stored as string
                            file_paths = [file_paths]
                    except json.JSONDecodeError:
                        file_paths = [file_paths_raw] if file_paths_raw else []

                    # Fallback for format_type if still missing
                    if not format_type and file_paths:
                        fp_ext = Path(file_paths[0]).suffix.lstrip('.')
                        if fp_ext:
                            format_type = fp_ext
                    
                    if not format_type or format_type == "None":
                        format_type = 'jsonl'
                    
                    # Handle multiple files (zip them)
                    if len(file_paths) > 1:
                        zip_buffer = io.BytesIO()
                        files_to_zip = []
                        for fp in file_paths:
                            fpath = Path(fp)
                            if not fpath.exists():
                                # Try fallback search for this batch file
                                logger.debug(f"Batch file {fp} not found, searching in storage/datasets...")
                                found = list(Path("storage/datasets").glob(f"**/{fpath.name}"))
                                if found:
                                    fpath = found[0]
                            if fpath.exists():
                                files_to_zip.append(fpath)
                        
                        if files_to_zip:
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                for fpath in files_to_zip:
                                    zip_file.write(fpath, arcname=fpath.name)
                            
                            zip_buffer.seek(0)
                            return StreamingResponse(
                                iter([zip_buffer.getvalue()]),
                                media_type="application/zip",
                                headers={"Content-Disposition": f"attachment; filename={name}_v{version}.zip"}
                            )
                    
                    # Handle single file
                    elif len(file_paths) == 1:
                        dataset_file = Path(file_paths[0])
                        logger.debug(f"Single file path from DB: {dataset_file}, exists: {dataset_file.exists()}")
                        
                        # If path doesn't exist, try resolving via search
                        if not dataset_file.exists():
                            filename_only = dataset_file.name
                            logger.debug(f"File not found at {dataset_file}, searching for {filename_only} in storage/datasets")
                            found = list(Path("storage/datasets").glob(f"**/{filename_only}"))
                            if found:
                                dataset_file = found[0]
                                logger.debug(f"Fallback found file at: {dataset_file}")
                        
                        download_filename = f"{name}_v{version}.{format_type}"
                        if format_type in ['json', 'jsonl']:
                            media_type = "application/json"
                        
            except Exception as db_err:
                logger.warning(f"Database lookup failed for download {dataset_id}: {db_err}")

        # 2. Fallback to Filesystem Search (Legacy/Filename lookup)
        if not dataset_file:
            datasets_base = Path("storage/datasets")
            if datasets_base.exists():
                for project_dir in datasets_base.iterdir():
                    if project_dir.is_dir():
                        potential_file = project_dir / dataset_id
                        if potential_file.exists():
                            dataset_file = potential_file
                            break

        if not dataset_file or not dataset_file.exists():
            raise HTTPException(status_code=404, detail=f"Dataset file {dataset_id} not found")

        return FileResponse(
            path=str(dataset_file),
            filename=download_filename,
            media_type=media_type
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download dataset: {str(e)}")


# Dataset Management Endpoints

class DatasetListItem(BaseModel):
    """Dataset list item model."""
    id: str
    name: str
    format_type: str
    size_bytes: int
    entries_count: int
    created_at: datetime
    status: str
    generation_params: Optional[dict] = None

class DatasetListResponse(BaseModel):
    """Dataset list response model."""
    datasets: List[DatasetListItem]
    total_count: int
    page: int
    page_size: int

class DatasetDetailsResponse(BaseModel):
    """Dataset details response model."""
    id: str
    name: str
    format_type: str
    size_bytes: int
    entries_count: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    file_path: Optional[str] = None
    generation_params: Optional[dict] = None
    statistics: Optional[dict] = None

class BulkDeleteRequest(BaseModel):
    """Bulk delete request model."""
    dataset_ids: List[str]

class BulkDeleteResponse(BaseModel):
    """Bulk delete response model."""
    deleted_count: int
    failed_deletions: List[dict]


@router.get("/list/{project_id}", response_model=DatasetListResponse)
async def list_datasets(
    project_id: str,
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    format_filter: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db=Depends(get_db)
 ):
    """List datasets for a project with filtering and pagination."""
    try:
        logger.debug(f"list_datasets called with project_id={project_id}, page={page}, page_size={page_size}")

        # Validate project exists
        project_repo = ProjectRepository(db)
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ?", (project_id,))
        project_row = cursor.fetchone()
        if not project_row:
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        # Get datasets from filesystem instead of database
        import os
        from pathlib import Path

        datasets_dir = Path("storage/datasets") / project_id
        logger.debug(f"Looking for datasets in {datasets_dir}")

        if not datasets_dir.exists():
            logger.debug(f"Dataset directory does not exist: {datasets_dir}")
            return DatasetListResponse(
                datasets=[],
                total_count=0,
                page=page,
                page_size=page_size
            )

        logger.debug(f"Dataset directory exists: {datasets_dir}")

        # Get all dataset files
        dataset_files = []
        for file_path in datasets_dir.glob("*"):
            if file_path.is_file():
                dataset_files.append(file_path)

        logger.debug(f"Found {len(dataset_files)} total files in {datasets_dir}")
        for file_path in dataset_files:
            logger.debug(f"File: {file_path.name}")

        # Get plugin format metadata for extension mapping
        plugin_format_extensions = {}
        try:
            from ...features.plugin.manager import plugin_manager
            for format_name in plugin_manager.get_extensions("compileo.datasetgen.formatter").keys():
                metadata = plugin_manager.get_format_metadata(format_name)
                if metadata and 'file_extension' in metadata:
                    plugin_format_extensions[metadata['file_extension']] = format_name
        except Exception:
            pass

        # Apply filters
        filtered_files = []
        for file_path in dataset_files:
            # Extract format from filename
            filename = file_path.name
            if filename.endswith('.json'):
                format_type = 'jsonl'  # JSON files are typically jsonl format
            elif filename.endswith('.parquet'):
                format_type = 'parquet'
            else:
                # Check if this file extension matches a plugin format
                file_extension = filename.split('.')[-1] if '.' in filename else ''
                if file_extension in plugin_format_extensions:
                    format_type = plugin_format_extensions[file_extension]
                else:
                    format_type = 'unknown'

            logger.debug(f"Processing file {filename}, format: {format_type}")

            # Apply format filter
            if format_filter and format_type != format_filter:
                logger.debug(f"Skipping {filename} due to format filter")
                continue

            # Apply search filter (search in filename)
            if search and search.lower() not in filename.lower():
                logger.debug(f"Skipping {filename} due to search filter")
                continue

            # Get file stats
            stat = file_path.stat()
            size_bytes = stat.st_size
            created_at = datetime.fromtimestamp(stat.st_ctime)

            logger.debug(f"File {filename} - size: {size_bytes}, created: {created_at}")

            # Apply date filters
            if date_from and created_at < date_from:
                logger.debug(f"Skipping {filename} due to date_from filter")
                continue
            if date_to and created_at > date_to:
                logger.debug(f"Skipping {filename} due to date_to filter")
                continue

            filtered_files.append({
                'file_path': file_path,
                'filename': filename,
                'format_type': format_type,
                'size_bytes': size_bytes,
                'created_at': created_at
            })

        logger.debug(f"After filtering, {len(filtered_files)} files remain")

        # Sort by creation date (newest first)
        filtered_files.sort(key=lambda x: x['created_at'], reverse=True)

        # Apply pagination
        total_count = len(filtered_files)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_files = filtered_files[start_idx:end_idx]

        # Convert to response model
        datasets = []
        for file_info in paginated_files:
            logger.debug(f"Processing file for response: {file_info['filename']}")

            # Check if this file is tracked in database (dataset_versions)
            db_id = file_info['filename'] # Default ID is filename
            db_name = file_info['filename']
            db_params = None
            db_count = 0
            
            try:
                cursor = db.cursor()
                # Search for filename in file_path column
                cursor.execute("SELECT id, dataset_name, total_entries, metadata FROM dataset_versions WHERE file_path LIKE ?", (f"%{file_info['filename']}%",))
                db_row = cursor.fetchone()
                if db_row:
                    db_id = db_row[0]
                    db_name = db_row[1]
                    db_count = db_row[2] or 0
                    metadata = json.loads(db_row[3]) if db_row[3] else {}
                    db_params = metadata.get('generation_params')
                    logger.debug(f"Found DB record for {file_info['filename']}: ID={db_id}")
            except Exception as e:
                logger.debug(f"Error checking DB for dataset file: {e}")

            # Try to get entry count from file content if DB count is 0
            entries_count = db_count
            if entries_count == 0:
                try:
                    if file_info['format_type'] == 'jsonl':
                        with open(file_info['file_path'], 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                # Count lines for jsonl format
                                entries_count = len([line for line in content.split('\n') if line.strip()])
                except Exception as e:
                    logger.debug(f"Error counting entries for {file_info['filename']}: {e}")
                    entries_count = 0

            dataset_item = DatasetListItem(
                id=str(db_id),  # Use UUID if found, otherwise filename
                name=db_name,
                format_type=file_info['format_type'],
                size_bytes=file_info['size_bytes'],
                entries_count=entries_count,
                created_at=file_info['created_at'],
                status="completed",  # All filesystem datasets are completed
                generation_params=db_params
            )
            datasets.append(dataset_item)
            logger.debug(f"Added dataset: {dataset_item.name} (ID: {dataset_item.id})")

        response = DatasetListResponse(
            datasets=datasets,
            total_count=total_count,
            page=page,
            page_size=page_size
        )

        logger.debug(f"Returning response with {len(datasets)} datasets, total_count={total_count}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get("/{dataset_id}/details", response_model=DatasetDetailsResponse)
async def get_dataset_details(dataset_id: str, db=Depends(get_db)):
    """Get detailed information about a specific dataset."""
    try:
        cursor = db.cursor()
        cursor.execute("""
            SELECT id, dataset_name as name, format_type, size_bytes, total_entries as entries_count, created_at, completed_at,
                   status, file_path, generation_params, NULL as statistics
            FROM dataset_versions
            WHERE id = ?
        """, (dataset_id,))

        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        return DatasetDetailsResponse(
            id=row[0],
            name=row[1],
            format_type=row[2],
            size_bytes=row[3] or 0,
            entries_count=row[4] or 0,
            created_at=row[5],
            completed_at=row[6],
            status=row[7] or "unknown",
            file_path=row[8],
            generation_params=json.loads(row[9]) if row[9] else None,
            statistics=json.loads(row[10]) if row[10] else None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset details: {str(e)}")


@router.delete("/{dataset_id}/delete")
async def delete_dataset(dataset_id: str, db=Depends(get_db)):
    """Delete a dataset and its associated files."""
    try:
        logger.debug(f"Attempting to delete dataset: {dataset_id}")

        # Support both UUID/ID lookup and direct filename lookup
        import os
        from pathlib import Path

        dataset_file = None
        file_paths_to_delete = []

        # 1. Try DB lookup first (if it's a UUID/ID)
        is_filename = dataset_id.endswith(('.json', '.jsonl', '.parquet'))
        if dataset_id != "None" and not is_filename:
            try:
                cursor = db.cursor()
                cursor.execute("SELECT file_path FROM dataset_versions WHERE id = ?", (dataset_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    try:
                        fps = json.loads(row[0])
                        file_paths_to_delete = fps if isinstance(fps, list) else [fps]
                    except:
                        file_paths_to_delete = [row[0]]
            except Exception as e:
                logger.debug(f"DB lookup failed for delete: {e}")

        # 2. Fallback to Filesystem Search (Legacy/Filename)
        if not file_paths_to_delete:
            datasets_base = Path("storage/datasets")
            if datasets_base.exists():
                for project_dir in datasets_base.iterdir():
                    if project_dir.is_dir():
                        potential_file = project_dir / dataset_id
                        if potential_file.exists():
                            file_paths_to_delete = [str(potential_file)]
                            break

        if not file_paths_to_delete:
            raise HTTPException(status_code=404, detail=f"Dataset file {dataset_id} not found")

        logger.debug(f"Found {len(file_paths_to_delete)} files to delete for dataset {dataset_id}")

        # Delete the files
        for fp_str in file_paths_to_delete:
            fp = Path(fp_str)
            if fp.exists():
                fp.unlink()
                logger.debug(f"Deleted file: {fp}")
            else:
                # Try fallback search for the filename part
                found = list(Path("storage/datasets").glob(f"**/{fp.name}"))
                if found:
                    found[0].unlink()
                    logger.debug(f"Deleted fallback file: {found[0]}")


        logger.debug(f"Proceeding with deletion of files for dataset {dataset_id}")


        # Also try to clean up any related database records if they exist
        # This handles the case where datasets might be tracked in dataset_versions table
        try:
            cursor = db.cursor()

            # First, get the dataset_versions entry to extract generation parameters
            cursor.execute("SELECT metadata FROM dataset_versions WHERE file_path LIKE ?", (f"%{dataset_id}%",))
            row = cursor.fetchone()

            if row and row[0]:
                try:
                    metadata = json.loads(row[0])
                    generation_params = metadata.get('generation_params', {})

                    # Delete matching dataset_parameters entry
                    # Match based on key generation parameters
                    cursor.execute("""
                        DELETE FROM dataset_parameters
                        WHERE project_id = ? AND purpose = ? AND audience = ? AND custom_audience = ? AND custom_purpose = ?
                    """, (
                        generation_params.get('project_id'),
                        generation_params.get('purpose') or '',
                        generation_params.get('audience') or '',
                        generation_params.get('custom_audience') or '',
                        generation_params.get('custom_purpose') or ''
                    ))
                    param_deleted = cursor.rowcount
                    if param_deleted > 0:
                        logger.debug(f"Cleaned up {param_deleted} dataset parameter records for dataset {dataset_id}")

                except (json.JSONDecodeError, KeyError) as json_error:
                    logger.warning(f"Failed to parse metadata for dataset {dataset_id}: {json_error}")

            # Delete dataset_versions entry
            cursor.execute("DELETE FROM dataset_versions WHERE file_path LIKE ?", (f"%{dataset_id}%",))
            version_deleted = cursor.rowcount
            if version_deleted > 0:
                logger.debug(f"Cleaned up {version_deleted} dataset version records for dataset {dataset_id}")

            db.commit()
        except Exception as db_error:
            logger.warning(f"Failed to clean up database records: {db_error}")
            # Don't fail the whole operation if DB cleanup fails

        return {"message": f"Dataset {dataset_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@router.post("/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_datasets(request: BulkDeleteRequest, db=Depends(get_db)):
    """Delete multiple datasets in bulk."""
    try:
        cursor = db.cursor()
        deleted_count = 0
        failed_deletions = []

        for dataset_id in request.dataset_ids:
            try:
                # Get dataset information
                cursor.execute("SELECT file_path FROM dataset_versions WHERE id = ?", (dataset_id,))
                row = cursor.fetchone()

                if row and row[0]:
                    # Delete file
                    try:
                        import os
                        if os.path.exists(row[0]):
                            os.remove(row[0])
                    except Exception as file_error:
                        logger.warning(f"Failed to delete file for dataset {dataset_id}: {file_error}")

                # Delete database record
                cursor.execute("DELETE FROM dataset_versions WHERE id = ?", (dataset_id,))
                deleted_count += 1

            except Exception as e:
                failed_deletions.append({
                    "dataset_id": dataset_id,
                    "error": str(e)
                })

        db.commit()

        return BulkDeleteResponse(
            deleted_count=deleted_count,
            failed_deletions=failed_deletions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform bulk delete: {str(e)}")


def process_evaluation_dataset_generation(job_id: str, request: DatasetGenerateRequest, db):
    """Background task for evaluation dataset generation."""
    try:
        job_storage[job_id].status = "running"
        job_storage[job_id].current_step = "Initializing evaluation datasets"

        # TODO: Implement evaluation dataset generation
        # Similar to regular generation but creates multiple evaluation sets

        import time
        time.sleep(3)
        job_storage[job_id].progress = 30
        job_storage[job_id].current_step = "Generating train/validation/test splits"

        time.sleep(3)
        job_storage[job_id].progress = 60
        job_storage[job_id].current_step = "Creating difficulty-based subsets"

        time.sleep(3)
        job_storage[job_id].progress = 90
        job_storage[job_id].current_step = "Finalizing evaluation datasets"

        time.sleep(2)
        job_storage[job_id].progress = 100
        job_storage[job_id].status = "completed"
        job_storage[job_id].current_step = "Evaluation datasets ready"
        job_storage[job_id].result = {
            "datasets": ["train.jsonl", "validation.jsonl", "test.jsonl", "difficulty_easy.jsonl", "difficulty_hard.jsonl"]
        }

    except Exception as e:
        job_storage[job_id].status = "failed"
        job_storage[job_id].error = str(e)


@router.post("/parameters", response_model=dict)
async def save_dataset_parameters(
    request: DatasetParameterRequest,
    db=Depends(get_db)
):
    """Save dataset parameters for a project."""
    try:
        # Validate project exists
        project_repo = ProjectRepository(db)
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ?", (request.project_id,))
        project_row = cursor.fetchone()
        if not project_row:
            raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")

        # Always create new parameters for each dataset generation
        # This maintains a history of all generation parameters
        dataset_param_repo = DatasetParameterRepository(db)
        param_id = dataset_param_repo.create({
            "id": str(uuid.uuid4()),
            "project_id": request.project_id,
            "purpose": request.purpose,
            "audience": request.audience,
            "extraction_rules": request.extraction_rules,
            "dataset_format": request.dataset_format,
            "question_style": request.question_style,
            "answer_style": request.answer_style,
            "negativity_ratio": request.negativity_ratio,
            "data_augmentation": request.data_augmentation,
            "custom_audience": request.custom_audience,
            "custom_purpose": request.custom_purpose,
            "complexity_level": request.complexity_level,
            "domain": request.domain,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })

        return {
            "message": "Dataset parameters saved successfully",
            "parameter_id": param_id,
            "project_id": request.project_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save dataset parameters: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save dataset parameters: {str(e)}")

@router.get("/parameters/{project_id}", response_model=dict)
async def get_latest_dataset_parameters(
    project_id: str,
    db=Depends(get_db)
):
    """Get the latest dataset parameters for a project."""
    try:
        from ...storage.src.project.database_repositories import DatasetParameterRepository
        repo = DatasetParameterRepository(db)
        params = repo.get_parameters_by_project(project_id)
        
        if not params:
            return {"parameters": None}
            
        return {"parameters": params[0]}  # Return the latest one

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset parameters: {str(e)}")

@router.get("/config/high-level-prompts", response_model=HighLevelPromptsResponse)
async def get_high_level_prompts_config():
    """Get high-level prompt configuration defaults and options."""
    return HighLevelPromptsResponse(
        audience_defaults=["healthcare professionals", "students", "patients", "researchers"],
        purpose_defaults=["patient education", "research", "medical education", "clinical decision support"],
        complexity_options=["beginner", "intermediate", "advanced", "expert"],
        domain_defaults=["general", "preventive nutrition", "cardiology", "pediatrics", "internal medicine"]
    )

@router.get("/config/default-prompts", response_model=DefaultPromptsResponse)
async def get_default_prompts_config():
    """Get default prompts for each generation mode."""
    return DefaultPromptsResponse(
        prompts={
            "instruction following": "Generate an instruction-response pair based on the following text. Create a natural language instruction that could be given to an AI assistant, and provide the appropriate response.\n\nText: {chunk}\n\nInstruction:",
            "question and answer": "Generate a question and answer pair based on the following text.\n\nText: {chunk}\n\nQuestion:",
            "question": "Generate a question based on the following text.\n\nText: {chunk}\n\nQuestion:",
            "answer": "Provide a comprehensive answer based on the following text.\n\nText: {chunk}\n\nAnswer:",
            "summarization": "Analyze the following text and provide concise summaries for different aspects or sections. If the text contains clear sections with headers (like # Section Title), summarize each section individually. If no clear sections exist, identify the key topics or aspects within the text and provide a separate summary for each topic.\n\nText: {chunk}\n\nSummaries:"
        }
    )
