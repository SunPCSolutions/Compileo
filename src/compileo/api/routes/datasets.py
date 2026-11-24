"""Datasets routes for the Compileo API."""

import os
import uuid
import json
import logging
from typing import List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import (
    ProjectRepository, DatasetParameterRepository, PromptRepository, ChunkRepository,
    DatasetJobRepository, DocumentRepository, ExtractionResultRepository, TaxonomyRepository,
    ExtractionJobRepository
)
from ..core.database import get_db
from ..core.dependencies import get_current_active_user
from ..models.user import User as UserSchema

# Create router
router = APIRouter()

# Global job storage for status tracking (in-memory fallback)
job_storage = {}

# Pydantic models
class DatasetGenerateRequest(BaseModel):
    project_id: str
    prompt_name: str = "default"
    custom_prompt: Optional[str] = None
    generation_mode: str = "default"
    format_type: str = "jsonl"
    concurrency: int = 1
    batch_size: int = 50
    include_evaluation_sets: bool = False
    taxonomy_project: Optional[str] = None
    taxonomy_name: Optional[str] = None
    data_source: str = "Chunks Only"
    extraction_file_id: Optional[str] = None
    selected_categories: Optional[List[str]] = None
    only_validated: bool = False
    output_dir: str = "."
    analyze_quality: bool = True
    quality_threshold: float = 0.7
    enable_versioning: bool = False
    dataset_name: Optional[str] = None
    run_benchmarks: bool = False
    benchmark_suite: str = "glue"
    parsing_model: str = "gemini"
    chunking_model: str = "gemini"
    classification_model: str = "gemini"
    datasets_per_chunk: int = Field(
        default=3,
        ge=0,
        description="Number of entries to generate per chunk. Set to 0 to let AI decide based on content."
    )
    purpose: Optional[str] = None
    
    @validator('datasets_per_chunk')
    def validate_datasets_per_chunk(cls, v):
        if v < 0:
            raise ValueError('datasets_per_chunk must be >= 0 (0 = AI decides)')
        return v
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
    status: str
    progress: int
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

class DatasetListItem(BaseModel):
    id: str
    name: str
    format_type: str
    size_bytes: int
    entries_count: int
    created_at: datetime
    status: str

class DatasetListResponse(BaseModel):
    datasets: List[DatasetListItem]
    total_count: int
    page: int
    page_size: int

class DatasetResponse(BaseModel):
    id: str
    name: str
    entries: List[DatasetEntry]
    total_entries: int
    created_at: datetime
    format_type: str
    quality_summary: Optional[dict] = None

class ExtractionFile(BaseModel):
    id: str
    taxonomy_id: Optional[str]
    completed_at: Optional[datetime]
    status: str
    categories_count: int
    display_name: str

class ExtractionFilesResponse(BaseModel):
    extraction_files: List[ExtractionFile]

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
    custom_audience: str = ""
    custom_purpose: str = ""
    complexity_level: str = "intermediate"
    domain: str = "general"

def get_repositories(
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    return {
        'dataset_job': DatasetJobRepository(current_user.id),
        'dataset_param': DatasetParameterRepository(current_user.id),
        'project': ProjectRepository(current_user.id),
        'document': DocumentRepository(current_user.id),
        'chunk': ChunkRepository(current_user.id),
        'extraction_result': ExtractionResultRepository(current_user.id),
        'taxonomy': TaxonomyRepository(current_user.id),
        'extraction_job': ExtractionJobRepository(current_user.id),
        'prompt': PromptRepository(current_user.id)
    }

@router.post("/generate", response_model=JobResponse)
async def generate_dataset(
    request: DatasetGenerateRequest,
    background_tasks: BackgroundTasks,
    repos=Depends(get_repositories),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Generate a dataset from processed chunks for current user."""
    from ...core.settings import BackendSettings
    user_settings = BackendSettings(owner_id=current_user.id)
    try:
        project = repos['project'].get_project_by_id(request.project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")
        project_name = project['name']

        # Resolve models from database settings once here
        parsing_model = (
            user_settings.get_parsing_gemini_model() if request.parsing_model == "gemini" else
            user_settings.get_parsing_grok_model() if request.parsing_model == "grok" else
            user_settings.get_parsing_openai_model() if request.parsing_model == "openai" else
            user_settings.get_parsing_ollama_model() if request.parsing_model == "ollama" else
            request.parsing_model
        )
        chunking_model = (
            user_settings.get_chunking_gemini_model() if request.chunking_model == "gemini" else
            user_settings.get_chunking_grok_model() if request.chunking_model == "grok" else
            user_settings.get_chunking_openai_model() if request.chunking_model == "openai" else
            user_settings.get_chunking_ollama_model() if request.chunking_model == "ollama" else
            request.chunking_model
        )
        classification_model = (
            user_settings.get_classification_gemini_model() if request.classification_model == "gemini" else
            user_settings.get_classification_grok_model() if request.classification_model == "grok" else
            user_settings.get_classification_openai_model() if request.classification_model == "openai" else
            user_settings.get_classification_ollama_model() if request.classification_model == "ollama" else
            request.classification_model
        )
        generation_model = (
            user_settings.get_generation_gemini_model() if request.classification_model == "gemini" else
            user_settings.get_generation_grok_model() if request.classification_model == "grok" else
            user_settings.get_generation_openai_model() if request.classification_model == "openai" else
            user_settings.get_generation_ollama_model() if request.classification_model == "ollama" else
            None
        )

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
        # Save dataset parameters using repository directly
        repos['dataset_param'].create({
            "id": str(uuid.uuid4()),
            "owner_id": current_user.id,
            "project_id": request.project_id,
            "purpose": param_request.purpose,
            "audience": param_request.audience,
            "extraction_rules": param_request.extraction_rules,
            "dataset_format": param_request.dataset_format,
            "question_style": param_request.question_style,
            "answer_style": param_request.answer_style,
            "negativity_ratio": param_request.negativity_ratio,
            "data_augmentation": param_request.data_augmentation,
            "custom_audience": param_request.custom_audience,
            "custom_purpose": param_request.custom_purpose,
            "complexity_level": param_request.complexity_level,
            "domain": param_request.domain,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })

        # Validate taxonomy if specified
        if request.taxonomy_project and request.taxonomy_name:
            # TODO: Implement taxonomy validation
            pass

        # Validate prompt exists only if no custom prompt is provided
        if not request.custom_prompt or request.custom_prompt.strip() == "":
            if request.prompt_name != "default" or not (request.taxonomy_project and request.taxonomy_name):
                prompt_repo = repos['prompt']
                prompt = prompt_repo.get_by_name(request.prompt_name, request.project_id)
                if not prompt:
                    if request.prompt_name != "default":
                        raise HTTPException(status_code=404, detail=f"Prompt '{request.prompt_name}' not found in project {request.project_id}")

        # Create job
        job_id = str(uuid.uuid4())
        
        # Initialize in-memory storage for status endpoint
        job_storage[job_id] = JobStatus(
            job_id=job_id,
            status="pending",
            progress=0,
            current_step="Initializing"
        )
        job_params = {
            'project_id': request.project_id,
            'owner_id': current_user.id,
            'generation_mode': request.generation_mode,
            'format_type': request.format_type,
            'data_source': request.data_source,
            'datasets_per_chunk': request.datasets_per_chunk,
            'concurrency': request.concurrency,
            'batch_size': request.batch_size,
            'custom_prompt': request.custom_prompt,
            'taxonomy_project': request.taxonomy_project,
            'taxonomy_name': request.taxonomy_name,
            'extraction_file_id': request.extraction_file_id,
            'selected_categories': request.selected_categories,
            'only_validated': request.only_validated,
            # Propagate specific model names
            'parsing_provider': request.parsing_model,
            'parsing_model': parsing_model,
            'chunking_provider': request.chunking_model,
            'chunking_model': chunking_model,
            'classification_provider': request.classification_model,
            'classification_model': classification_model,
            'generation_model': generation_model
        }

        job_data = {
            'id': job_id,
            'owner_id': current_user.id,
            'project_id': request.project_id,
            'job_type': 'dataset_generation',
            'status': 'pending',
            'parameters': job_params,
            'progress': 0.0,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        repos['dataset_job'].create_job(job_data)

        from ...features.jobhandle.enhanced_job_queue import enhanced_job_queue_manager, JobType, JobPriority
        if enhanced_job_queue_manager:
            enhanced_job_queue_manager.submit_job(
                job_type=JobType.DATASET_GENERATION,
                parameters=job_params,
                priority=JobPriority.NORMAL,
                user_id=current_user.id,
                job_id=job_id
            )
        else:
            # Fallback to background tasks if Redis is disabled
            background_tasks.add_task(
                process_dataset_generation,
                job_id,
                request,
                current_user.id,
                project_name,
                parsing_model,
                chunking_model,
                classification_model,
                generation_model
            )

        return JobResponse(job_id=job_id, message="Dataset generation started")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-evaluation", response_model=JobResponse)
async def generate_evaluation_dataset(
    request: DatasetGenerateRequest,
    background_tasks: BackgroundTasks,
    repos=Depends(get_repositories),
    current_user: UserSchema = Depends(get_current_active_user)
):
    try:
        project = repos['project'].get_project_by_id(request.project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")

        job_id = str(uuid.uuid4())
        job_data = {
            'id': job_id,
            'owner_id': current_user.id,
            'project_id': request.project_id,
            'status': 'pending',
            'parameters': request.dict(),
            'progress': 0.0,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        repos['dataset_job'].create_job(job_data)

        background_tasks.add_task(
            process_evaluation_dataset_generation,
            job_id,
            request,
            current_user.id
        )

        return JobResponse(job_id=job_id, message="Evaluation dataset generation started")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate/{job_id}/status", response_model=JobStatus)
async def get_generation_status(
    job_id: str,
    repos=Depends(get_repositories),
    current_user: UserSchema = Depends(get_current_active_user)
):
    if job_id in job_storage:
        return job_storage[job_id]
        
    job = repos['dataset_job'].get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(
        job_id=job['id'],
        status=job['status'],
        progress=int(job['progress'] * 100) if job['progress'] else 0,
        current_step=job['status'].capitalize(),
        result=json.loads(job['result']) if job.get('result') else None,
        error=job.get('error')
    )

@router.get("/extraction-files/{project_id}", response_model=ExtractionFilesResponse)
async def get_extraction_files(
    project_id: str,
    repos=Depends(get_repositories),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Returns list of completed extraction jobs for a project."""
    try:
        # Verify project ownership
        project = repos['project'].get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get completed extraction jobs for the project
        jobs = repos['extraction_job'].get_jobs_by_project(project_id)
        
        extraction_files = []
        for job in jobs:
            if job['status'] == 'completed':
                # Parse parameters to get category counts if available
                params = job.get('parameters', {})
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except:
                        params = {}
                
                selected_categories = params.get('selected_categories', [])
                
                extraction_files.append(ExtractionFile(
                    id=job['id'],
                    taxonomy_id=params.get('taxonomy_id'),
                    completed_at=job['completed_at'],
                    status=job['status'],
                    categories_count=len(selected_categories) if selected_categories else 0,
                    display_name=f"Job {job['id'][:8]} - {params.get('extraction_type', 'unknown')} ({len(selected_categories) if selected_categories else 0} categories)"
                ))
                
        return ExtractionFilesResponse(extraction_files=extraction_files)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list/{project_id}", response_model=DatasetListResponse)
async def list_datasets(
    project_id: str,
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    format_filter: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
 ):
    try:
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?", (project_id, current_user.id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Project not found")

        from ...storage.src.project.file_manager import FileManager
        datasets_dir = FileManager.get_storage_base_path() / current_user.id / "datasets" / project_id
        
        if not datasets_dir.exists():
            return DatasetListResponse(datasets=[], total_count=0, page=page, page_size=page_size)

        files = [f for f in datasets_dir.glob("*") if f.is_file()]
        filtered = []
        for f in files:
            stat = f.stat()
            created_at = datetime.fromtimestamp(stat.st_ctime)
            
            # Count entries in the dataset file
            entries_count = 0
            try:
                if f.suffix == '.json' or f.name.endswith('.jsonl'):
                    with open(f, 'r', encoding='utf-8') as ef:
                        for line in ef:
                            line = line.strip()
                            if line:
                                entries_count += 1
                elif f.suffix == '.parquet':
                    import pandas as pd
                    df = pd.read_parquet(f)
                    entries_count = len(df)
            except Exception:
                pass  # Keep entries_count at 0 if counting fails
            
            filtered.append(DatasetListItem(
                id=f.name,
                name=f.name,
                format_type="jsonl" if f.name.endswith(".json") else "parquet",
                size_bytes=stat.st_size,
                entries_count=entries_count,
                created_at=created_at,
                status="completed"
            ))
        
        return DatasetListResponse(
            datasets=filtered[(page-1)*page_size:page*page_size],
            total_count=len(filtered),
            page=page,
            page_size=page_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/{filename}/download")
async def download_dataset_file(
    project_id: str,
    filename: str,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Download a dataset file directly."""
    from fastapi.responses import FileResponse
    from ...storage.src.project.file_manager import FileManager
    file_path = FileManager.get_storage_base_path() / current_user.id / "datasets" / project_id / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Dataset file not found")
        
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

@router.get("/{dataset_id}/download")
async def get_dataset_download_url(
    dataset_id: str,
    project_id: Optional[str] = None,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Legacy endpoint to support existing UI that expects a download URL.
    Returns the first matching file for the dataset id (job id) or exact filename."""
    from ...storage.src.project.file_manager import FileManager
    
    # If project_id is missing, we can't easily construct the path
    # unless we search all projects, which is slow.
    if project_id:
        datasets_dir = FileManager.get_storage_base_path() / current_user.id / "datasets" / project_id
        if datasets_dir.exists():
            files = list(datasets_dir.glob(f"*{dataset_id}*"))
            if files:
                return {"download_url": f"/api/v1/datasets/{project_id}/{files[0].name}/download"}
                
    return {"download_url": None}

@router.get("/{dataset_id}/details")
async def get_dataset_details(
    dataset_id: str,
    project_id: Optional[str] = None,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get details of a specific dataset file."""
    from ...storage.src.project.file_manager import FileManager
    
    # Try to extract job_id from dataset_id (filename) to lookup project_id in DB
    # Format: dataset_{job_id}_batch_{idx}.ext
    resolved_project_id = project_id
    
    if not resolved_project_id:
        import re
        # Extract UUID from filename
        match = re.search(r'dataset_([a-f0-9\-]{36})', dataset_id)
        if match:
            job_id = match.group(1)
            # Lookup job to get project_id
            job_repo = DatasetJobRepository(current_user.id)
            job = job_repo.get_job_by_id(job_id)
            if job and job.get('project_id'):
                resolved_project_id = job.get('project_id')
    
    if not resolved_project_id:
        # Fallback: if we still don't have a project_id, we can't efficiently locate the file
        # without scanning everything, but the user requested to avoid scanning fallback if possible.
        # However, for robustness, if we can't extract ID, we might still fail or do a scan.
        # Given the instruction "project_id should always be provided", we'll enforce it or fail if not resolvable.
        raise HTTPException(status_code=400, detail="project_id is required or could not be resolved from dataset filename")

    datasets_dir = FileManager.get_storage_base_path() / current_user.id / "datasets" / resolved_project_id
    
    if not datasets_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project datasets directory not found for project {resolved_project_id}")
    
    # Find the dataset file (dataset_id might be partial match or full name)
    dataset_files = list(datasets_dir.glob(f"*{dataset_id}*"))
    if not dataset_files:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    dataset_file = dataset_files[0]
    
    if not dataset_file.is_file():
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    # Get file stats
    stat = dataset_file.stat()
    
    # Try to read sample records from the file
    sample_records = []
    record_count = 0
    try:
        if dataset_file.suffix == '.json' or dataset_file.name.endswith('.jsonl'):
            with open(dataset_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                record_count = len(lines)
                # Get first 5 records as samples
                for line in lines[:5]:
                    line = line.strip()
                    if line:
                        try:
                            sample_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        elif dataset_file.suffix == '.parquet':
            try:
                import pandas as pd
                df = pd.read_parquet(dataset_file)
                record_count = len(df)
                sample_records = df.head(5).to_dict(orient='records')
            except Exception:
                pass
    except Exception:
        pass
    
    return {
        "id": dataset_id,
        "name": dataset_file.name,
        "format_type": "parquet" if dataset_file.suffix == '.parquet' else "jsonl",
        "file_size": stat.st_size,
        "record_count": record_count,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "sample_records": sample_records,
        "chunking_strategy": "unknown",
        "embedding_model": "unknown"
    }


@router.delete("/{dataset_id}/delete")
async def delete_dataset(
    dataset_id: str,
    project_id: Optional[str] = None,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Delete a dataset file."""
    from ...storage.src.project.file_manager import FileManager
    
    # Try to extract job_id from dataset_id (filename) to lookup project_id in DB
    resolved_project_id = project_id
    
    if not resolved_project_id:
        import re
        match = re.search(r'dataset_([a-f0-9\-]{36})', dataset_id)
        if match:
            job_id = match.group(1)
            job_repo = DatasetJobRepository(current_user.id)
            job = job_repo.get_job_by_id(job_id)
            if job and job.get('project_id'):
                resolved_project_id = job.get('project_id')
    
    if not resolved_project_id:
        raise HTTPException(status_code=400, detail="project_id is required or could not be resolved from dataset filename")

    datasets_dir = FileManager.get_storage_base_path() / current_user.id / "datasets" / resolved_project_id
    
    if not datasets_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project datasets directory not found for project {resolved_project_id}")
    
    # Find the dataset file (dataset_id might be partial match)
    dataset_files = list(datasets_dir.glob(f"*{dataset_id}*"))
    if not dataset_files:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    # Delete all matching files
    deleted_count = 0
    for dataset_file in dataset_files:
        if dataset_file.is_file():
            dataset_file.unlink()
            deleted_count += 1
    
    return {"message": f"Successfully deleted {deleted_count} dataset file(s)"}


def process_dataset_generation(job_id: str, request: DatasetGenerateRequest, owner_id: str, project_name: str, parsing_model, chunking_model, classification_model, generation_model):
    """Background task for dataset generation with batch processing."""
    import json
    import os
    job_repo = None
    try:
        if job_id not in job_storage:
            job_storage[job_id] = JobStatus(
                job_id=job_id,
                status="pending",
                progress=0,
                current_step="Initializing"
            )

        job_storage[job_id].status = "running"
        job_storage[job_id].current_step = "Initializing dataset generation"
        
        # Initialize repository
        from ...storage.src.project.database_repositories import DatasetJobRepository
        job_repo = DatasetJobRepository(owner_id)
        job_repo.update_job_status(job_id, 'running')

        from ...features.datasetgen.generator import DatasetGenerator
        from ...features.datasetgen.prompt_builder import PromptBuilder
        from ...features.datasetgen.output_formatter import OutputFormatter
        from ...features.datasetgen.llm_interaction import LLMInteraction
        from ...features.taxonomy.loader import TaxonomyLoader
        from ...storage.src.project.database_repositories import (
            PromptRepository, DatasetParameterRepository, DocumentRepository,
            ChunkRepository, ExtractionResultRepository, TaxonomyRepository,
            DatasetJobRepository, DatasetVersionRepository, DatasetChangeRepository, DatasetLineageRepository,
            ExtractionJobRepository
        )
        from ...features.datasetgen.version_manager import DatasetVersionManager
        from ...storage.src.project.file_manager import FileManager

        # Initialize repositories
        prompt_repo = PromptRepository(owner_id)
        dataset_param_repo = DatasetParameterRepository(owner_id)
        document_repo = DocumentRepository(owner_id)
        chunk_repo = ChunkRepository(owner_id)
        extraction_result_repo = ExtractionResultRepository(owner_id)
        extraction_job_repo = ExtractionJobRepository(owner_id)
        version_repo = DatasetVersionRepository(owner_id)
        change_repo = DatasetChangeRepository(owner_id)
        lineage_repo = DatasetLineageRepository(owner_id)

        # Initialize core components
        prompt_builder = PromptBuilder(prompt_repo, dataset_param_repo)
        output_formatter = OutputFormatter()
        
        # Get API keys and resolve models
        from ...core.settings import BackendSettings
        user_settings = BackendSettings(owner_id=owner_id)
        
        llm_provider = request.classification_model or user_settings.get_generation_model()
        api_key = None
        if llm_provider == "grok":
            api_key = user_settings.get_grok_api_key()
        elif llm_provider == "gemini":
            api_key = user_settings.get_gemini_api_key()
        elif llm_provider == "openai":
            api_key = user_settings.get_openai_api_key()

        llm_interaction = LLMInteraction(
            llm_provider=llm_provider,
            api_key=api_key,
            model=generation_model or classification_model
        )

        taxonomy_loader = None
        if request.taxonomy_project and request.taxonomy_name:
            taxonomy_repo = TaxonomyRepository(owner_id)
            project_repo = ProjectRepository(owner_id)
            taxonomy_loader = TaxonomyLoader(taxonomy_repo, project_repo)

        version_manager = DatasetVersionManager(version_repo, change_repo, lineage_repo)

        dataset_generator = DatasetGenerator(
            prompt_builder=prompt_builder,
            output_formatter=output_formatter,
            llm_interaction=llm_interaction,
            document_repository=document_repo,
            extraction_result_repository=extraction_result_repo,
            extraction_job_repository=extraction_job_repo,
            taxonomy_loader=taxonomy_loader,
            version_manager=version_manager
        )

        job_storage[job_id].progress = 10
        job_storage[job_id].current_step = "Loading project chunks"
        job_repo.update_job_progress(job_id, 0.1)

        # Get chunks for the project from the database using repositories
        document_repo = DocumentRepository(owner_id)
        chunk_repo = ChunkRepository(owner_id)
        documents = document_repo.get_documents_by_project_id(request.project_id)

        chunks = []
        storage_base = FileManager.get_storage_base_path()
        for doc in documents:
            doc_chunks = chunk_repo.get_chunks_by_document(doc['id'])
            for chunk in doc_chunks:
                p = chunk.get('file_path')
                if not p:
                    continue
                if not os.path.isabs(p): p = storage_base / p
                if os.path.exists(p):
                    with open(p, 'r', encoding='utf-8') as f: chunks.append(f.read().strip())

        if not chunks: raise ValueError("No chunks found")

        job_storage[job_id].progress = 25
        job_storage[job_id].current_step = "Processing chunks in batches"
        job_repo.update_job_progress(job_id, 0.25)

        # Determine data source mode
        prefer_extracted = request.data_source == "Extract"
        
        # Batch processing logic
        total_chunks = len(chunks)
        batch_size = request.batch_size or 50
        num_batches = (total_chunks + batch_size - 1) // batch_size
        all_results = []
        batch_files = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_chunks)
            batch_chunks = chunks[start_idx:end_idx]

            batch_progress = int(25 + (batch_idx / num_batches) * 50)
            job_storage[job_id].progress = batch_progress
            job_storage[job_id].current_step = f"Processing batch {batch_idx + 1}/{num_batches}"
            job_repo.update_job_progress(job_id, batch_progress / 100.0)

            # Generate for this batch
            batch_result = dataset_generator.generate_dataset(
                project_id=request.project_id,
                owner_id=owner_id,
                chunks=batch_chunks,
                prompt_name=request.prompt_name,
                format_type="raw_list", # Get list for internal processing
                datasets_per_chunk=request.datasets_per_chunk,
                prefer_extracted=prefer_extracted,
                enable_versioning=False, # Handle versioning at the end
                custom_prompt=request.custom_prompt,
                extraction_file_id=request.extraction_file_id,
                generation_mode=request.generation_mode
            )
            
            if isinstance(batch_result, list):
                all_results.extend(batch_result)
                
                # Save batch file
                from ...storage.src.project.file_manager import FileManager
                file_manager = FileManager(storage_type='datasets')
                
                # Use correct extension based on format_type, checking plugin metadata
                ext = "parquet" if request.format_type == "parquet" else "json"
                try:
                    from ...features.plugin.manager import plugin_manager
                    metadata = plugin_manager.get_format_metadata(request.format_type)
                    if metadata and 'file_extension' in metadata:
                        ext = metadata['file_extension']
                except Exception:
                    pass

                batch_filename = f"dataset_{job_id}_batch_{batch_idx}.{ext}"
                
                # Format using OutputFormatter
                batch_content = output_formatter.format_dataset(batch_result, request.format_type, owner_id)
                if isinstance(batch_content, str):
                    batch_content = batch_content.encode('utf-8')
                
                batch_file_path = file_manager.store_file(
                    project_id=request.project_id,
                    document_id=None,
                    file_name=batch_filename,
                    file_content=batch_content, # type: ignore
                    owner_id=owner_id
                )
                
                batch_files.append({
                    "batch_index": batch_idx,
                    "file_path": str(batch_file_path),
                    "entries_count": len(batch_result)
                })

        # Complete the job and handle versioning
        job_storage[job_id].progress = 100
        job_storage[job_id].status = "completed"
        job_storage[job_id].current_step = "Dataset generation complete"
        
        final_result = {
            "dataset_id": f"dataset_{job_id}",
            "entries_count": len(all_results),
            "batch_files": batch_files,
            "total_batches": num_batches,
            "parsing_model": parsing_model,
            "chunking_model": chunking_model,
            "classification_model": classification_model,
            "generation_model": generation_model
        }
        job_storage[job_id].result = final_result
        
        job_repo.update_job_result(job_id, final_result)
        job_repo.update_job_status(job_id, 'completed')
        job_repo.update_job_progress(job_id, 1.0)
        
        if request.enable_versioning and version_manager:
            try:
                version_manager.create_new_version(
                    request.project_id, owner_id, request.dataset_name or f"dataset_{job_id}", 
                    all_results
                )
            except Exception as e:
                logging.warning(f"Failed to create dataset version: {e}")

    except Exception as e:
        import traceback
        logging.error(f"Background task failed: {e}")
        logging.error(traceback.format_exc())
        job_storage[job_id].status = "failed"
        job_storage[job_id].error = str(e)
        if job_repo:
            job_repo.update_job_status(job_id, 'failed', str(e))

def process_evaluation_dataset_generation(job_id: str, request: DatasetGenerateRequest, owner_id: str):
    try:
        dataset_job_repo = DatasetJobRepository(owner_id)
        dataset_job_repo.update_job_status(job_id, "running")

        from ...features.datasetgen.generator import DatasetGenerator
        from ...features.datasetgen.prompt_builder import PromptBuilder
        from ...features.datasetgen.output_formatter import OutputFormatter
        from ...features.datasetgen.llm_interaction import LLMInteraction
        from ...storage.src.project.database_repositories import PromptRepository, DatasetParameterRepository, DocumentRepository, ExtractionResultRepository, ProjectRepository
        
        from ...core.settings import BackendSettings
        user_settings = BackendSettings(owner_id=owner_id)

        llm_provider = request.classification_model or user_settings.get_generation_model()
        llm_model = None
        api_key = None
        if llm_provider == "grok":
            llm_model = user_settings.get_generation_grok_model()
            api_key = user_settings.get_grok_api_key()
        elif llm_provider == "gemini":
            llm_model = user_settings.get_generation_gemini_model()
            api_key = user_settings.get_gemini_api_key()
        elif llm_provider == "openai":
            llm_model = user_settings.get_generation_openai_model()
            api_key = user_settings.get_openai_api_key()

        dataset_generator = DatasetGenerator(
            prompt_builder=PromptBuilder(PromptRepository(owner_id), DatasetParameterRepository(owner_id)),
            output_formatter=OutputFormatter(),
            llm_interaction=LLMInteraction(llm_provider=llm_provider, api_key=api_key, model=llm_model),
            document_repository=DocumentRepository(owner_id),
            extraction_result_repository=ExtractionResultRepository(owner_id)
        )

        # Get chunks using repositories
        document_repo = DocumentRepository(owner_id)
        chunk_repo = ChunkRepository(owner_id)
        documents = document_repo.get_documents_by_project_id(request.project_id)

        chunk_texts = []
        for doc in documents:
            doc_chunks = chunk_repo.get_chunks_by_document(doc['id'])
            for chunk in doc_chunks:
                p = chunk.get('file_path')
                if not p:
                    continue
                if os.path.exists(p):
                    with open(p, 'r', encoding='utf-8') as f:
                        chunk_texts.append(f.read().strip())

        dataset_generator.generate_evaluation_dataset(
            project_id=request.project_id,
            owner_id=owner_id,
            chunks=chunk_texts,
            prompt_name=request.prompt_name,
            format_type=request.format_type,
            datasets_per_chunk=request.datasets_per_chunk
        )
        dataset_job_repo.update_job_status(job_id, "completed")
    except Exception as e:
        import traceback
        logging.error(f"Background task failed: {e}")
        logging.error(traceback.format_exc())
        dataset_job_repo = DatasetJobRepository(owner_id)
        dataset_job_repo.update_job_status(job_id, "failed", str(e))

@router.post("/parameters", response_model=dict)
async def save_dataset_parameters(
    request: DatasetParameterRequest,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    try:
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?", (request.project_id, current_user.id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Project not found")

        dataset_param_repo = DatasetParameterRepository(current_user.id)
        param_id = dataset_param_repo.create({
            "id": str(uuid.uuid4()),
            "owner_id": current_user.id,
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
        return {"parameter_id": param_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
