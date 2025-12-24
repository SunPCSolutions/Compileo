"""Extraction routes for the Compileo API."""

import json
import logging
import os
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, status, Request
from datetime import datetime, timedelta

from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import (
    TaxonomyRepository,
    ExtractionJobRepository,
    ExtractionResultRepository
)
from ...storage.src.project.file_manager import FileManager
from ...features.extraction.taxonomy_extractor import TaxonomyExtractor
from ...core.settings import backend_settings
from ...core.logging import get_logger
from ..models.extraction import (
    SelectiveExtractionRequest,
    ExtractionJobResponse,
    ExtractionJobStatus,
    ExtractionResultItem,
    ExtractionResultsResponse,
    ExtractionJobCancellation,
    ExtractionJobRestart
)
from ...features.extraction.exceptions import (
    TaxonomyNotFoundError,
    TaxonomyValidationError,
    TaxonomyCategoryError,
    ClassifierUnavailableError,
    ClassificationFailureError,
    APIConnectionError,
    APIAuthenticationError,
    APIRateLimitError,
    JobNotFoundError,
    JobStateError,
    DatabaseConnectionError,
    ExtractionError
)
from ...features.extraction.error_logging import api_logger
from ...features.extraction.filesystem_storage import (
    ExtractionFileManager,
    FilesystemStorageManager,
    HybridStorageManager
)
from ...features.extraction.storage_monitor import (
    get_metrics,
    get_alert_manager,
    get_monitor,
    initialize_monitoring,
    StorageHealthMonitor
)
from ...features.jobhandle.models import JobType

# Create router
router = APIRouter()
logger = get_logger(__name__)


def get_job_queue_manager(request):
    """Get the job queue manager from the app state."""
    return request.app.state.job_queue_manager


def handle_extraction_error(error: Exception, operation: str, **context) -> HTTPException:
    """
    Convert extraction errors to appropriate HTTP responses.

    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        **context: Additional context for logging

    Returns:
        HTTPException with appropriate status code and message
    """
    # Log the error
    api_logger.log_error(error, operation, context=context)

    # Map exceptions to HTTP status codes and user-friendly messages
    if isinstance(error, TaxonomyNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested taxonomy was not found. Please check the taxonomy name and project."
        )
    elif isinstance(error, TaxonomyValidationError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The taxonomy structure is invalid. Please verify the taxonomy is properly formatted."
        )
    elif isinstance(error, TaxonomyCategoryError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid category selection. Please check the selected categories exist in the taxonomy."
        )
    elif isinstance(error, ClassifierUnavailableError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification service is currently unavailable. Please try again later or contact support."
        )
    elif isinstance(error, (APIConnectionError, APIAuthenticationError)):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="External service connection failed. Please try again later."
        )
    elif isinstance(error, APIRateLimitError):
        retry_after = error.details.get('retry_after_seconds', '60')
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please wait {retry_after} seconds before retrying.",
            headers={"Retry-After": str(retry_after)}
        )
    elif isinstance(error, JobNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested extraction job was not found."
        )
    elif isinstance(error, JobStateError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is in an invalid state: {error.details.get('current_status', 'unknown')}."
        )
    elif isinstance(error, DatabaseConnectionError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is temporarily unavailable. Please try again later."
        )
    elif isinstance(error, ClassificationFailureError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content classification failed. Please try again or contact support if the issue persists."
        )
    elif isinstance(error, ExtractionError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An extraction error occurred. Please try again or contact support."
        )
    else:
        # Generic error for unexpected exceptions
        api_logger.log_error(
            error,
            "unhandled_error",
            context={**context, "error_type": type(error).__name__}
        )
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again or contact support."
        )

# Dependency to get database connection
def get_db():
    return get_db_connection()

# Dependency to get repositories
def get_repositories(db=Depends(get_db)):
    file_manager = ExtractionFileManager()
    filesystem_manager = FilesystemStorageManager(file_manager)
    return {
        'taxonomy_repo': TaxonomyRepository(db),
        'extraction_job': ExtractionJobRepository(db),
        'extraction_result': ExtractionResultRepository(db),
        'file_manager': FileManager(),
        'filesystem_manager': filesystem_manager,
        'hybrid_manager': HybridStorageManager(filesystem_manager, ExtractionResultRepository(db), enable_filesystem=True)
    }

# Dependency to get taxonomy extractor
def get_taxonomy_extractor(repos=Depends(get_repositories), db=Depends(get_db)):
    grok_key = backend_settings.get_setting("grok_api_key")
    gemini_key = backend_settings.get_setting("gemini_api_key")

    logger.debug(f"Grok key available: {grok_key is not None}")
    logger.debug(f"Gemini key available: {gemini_key is not None}")

    # Create extractor with all available keys - classifier will be set per request
    from ...storage.src.project.database_repositories import ChunkRepository
    chunk_repo = ChunkRepository(db)
    extractor = TaxonomyExtractor(
        taxonomy_repo=repos['taxonomy_repo'],
        chunk_repo=chunk_repo,
        file_manager=repos['file_manager'],
        grok_api_key=grok_key,
        gemini_api_key=gemini_key,
        ollama_available=False  # Disable Ollama to use API models
    )

    # Don't override classifiers here - let the request specify which one to use
    logger.debug(f"Extractor created with available classifiers: grok={grok_key is not None}, gemini={gemini_key is not None}")
    return extractor


@router.post("/", response_model=ExtractionJobResponse)
async def create_extraction_job(
    request: SelectiveExtractionRequest,
    http_request: Request,
    repos=Depends(get_repositories)
):
    """Create a new selective extraction job."""
    try:
        # Validate request parameters
        if not request.taxonomy_id or not isinstance(request.taxonomy_id, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid taxonomy_id: must be a valid string identifier"
            )

        if request.selected_categories and not isinstance(request.selected_categories, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid selected_categories: must be a list of strings"
            )

        # Validate taxonomy exists and get project info
        try:
            taxonomy_data = repos['taxonomy_repo'].get_taxonomy_by_id(request.taxonomy_id)
            if not taxonomy_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Taxonomy with ID {request.taxonomy_id} not found"
                )
            project_id = taxonomy_data['project_id']
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise handle_extraction_error(e, "validate_taxonomy", taxonomy_id=request.taxonomy_id)

        # Submit job to the enhanced queue
        try:
            # Get job queue manager from app state
            job_queue_manager = get_job_queue_manager(http_request)
            if job_queue_manager is None:
                raise ValueError("Job queue manager not initialized")

            # Get specific models for initial and validation classifiers from database settings
            initial_classifier = request.initial_classifier or backend_settings.get_classification_model()
            validation_classifier = request.validation_classifier or initial_classifier

            initial_model = None
            if initial_classifier == "gemini":
                initial_model = backend_settings.get_classification_gemini_model()
            elif initial_classifier == "grok":
                initial_model = backend_settings.get_classification_grok_model()
            elif initial_classifier == "openai":
                initial_model = backend_settings.get_classification_openai_model()
            elif initial_classifier == "ollama":
                initial_model = backend_settings.get_classification_ollama_model()

            validation_model = None
            if validation_classifier == "gemini":
                validation_model = backend_settings.get_classification_gemini_model()
            elif validation_classifier == "grok":
                validation_model = backend_settings.get_classification_grok_model()
            elif validation_classifier == "openai":
                validation_model = backend_settings.get_classification_openai_model()
            elif validation_classifier == "ollama":
                validation_model = backend_settings.get_classification_ollama_model()

            job_params = {
                "operation": "extraction",
                "taxonomy_id": request.taxonomy_id,
                "selected_categories": request.selected_categories,
                "parameters": request.parameters,
                "initial_classifier": initial_classifier,
                "initial_model": initial_model,
                "enable_validation_stage": request.enable_validation_stage,
                "validation_classifier": validation_classifier,
                "validation_model": validation_model,
                "extraction_type": request.extraction_type,
                "extraction_mode": request.extraction_mode,
            }

            # Check job parameters being created
            logger.debug(f"job_params keys = {list(job_params.keys())}")
            logger.debug(f"extraction_mode = {job_params.get('extraction_mode', 'NOT_SET')}")
            logger.debug(f"request.extraction_mode = {request.extraction_mode}")
            job_id = job_queue_manager.submit_job(
                job_type=JobType.EXTRACTION,
                parameters=job_params
            )

            # Create job record in database
            job_data = {
                'id': job_id,
                'project_id': project_id,
                'document_id': None,  # No specific document for taxonomy extraction
                'status': 'pending',
                'parameters': job_params,
                'progress': 0.0,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'started_at': None,
                'completed_at': None,
                'error_message': None
            }
            repos['extraction_job'].create_job(job_data)
        except Exception as e:
            raise handle_extraction_error(e, "submit_extraction_job", taxonomy_id=request.taxonomy_id)

        api_logger.log_operation_start(
            "create_extraction_job",
            context={"job_id": job_id, "taxonomy_id": request.taxonomy_id, "project_id": project_id}
        )

        return ExtractionJobResponse(
            job_id=job_id,
            status="pending",
            message="Extraction job created and queued for processing"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_extraction_error(e, "create_extraction_job", taxonomy_id=getattr(request, 'taxonomy_id', None))

@router.get("/{job_id}", response_model=ExtractionJobStatus)
async def get_extraction_job_status(
    job_id: str,
    repos=Depends(get_repositories)
):
    """Get the status of an extraction job."""
    try:
        # Validate job_id - accept both string (UUID) and integer formats
        if isinstance(job_id, str):
            # UUID string format
            import re
            if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', job_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid job_id: must be a valid UUID string or positive integer"
                )
        elif isinstance(job_id, int):
            # Integer format
            if job_id <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid job_id: must be a positive integer"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid job_id: must be a UUID string or positive integer"
            )

        # Get job with error handling
        try:
            job = repos['extraction_job'].get_job_by_id(job_id)
            if not job:
                raise JobNotFoundError(job_id)
        except JobNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extraction job with ID {job_id} not found"
            )
        except Exception as e:
            raise handle_extraction_error(e, "get_job_status", job_id=job_id)

        # Get result metadata if job is completed
        progress = None
        if job['status'] == 'completed':
            try:
                metadata = repos['hybrid_manager'].get_job_metadata(job_id, storage_preference="hybrid")
                if metadata:
                    progress = {
                        'total_results': metadata['total_results'],
                        'avg_confidence': metadata['avg_confidence'],
                        'categories_found': metadata['categories']
                    }
            except Exception as e:
                api_logger.log_warning(
                    f"Failed to get result metadata for completed job {job_id}: {e}",
                    "get_job_status",
                    context={"job_id": job_id, "job_status": job['status']}
                )
                # Fallback to database metadata
                try:
                    metadata = repos['extraction_result'].get_result_metadata(job_id)
                    progress = {
                        'total_results': metadata['total_results'],
                        'avg_confidence': metadata['avg_confidence'],
                        'categories_found': metadata['categories']
                    }
                except Exception:
                    # Don't fail the request, just return without progress details
                    pass

        # Extract taxonomy_id from parameters
        taxonomy_id = None
        if job.get('parameters'):
            try:
                params = json.loads(job['parameters']) if isinstance(job['parameters'], str) else job['parameters']
                taxonomy_id = params.get('taxonomy_id')
            except (json.JSONDecodeError, TypeError):
                pass

        return ExtractionJobStatus(
            job_id=job['id'],
            taxonomy_id=taxonomy_id,
            status=job['status'],
            progress_percentage=job['progress'],
            progress=progress,
            created_at=job['created_at'],
            updated_at=job['updated_at'],
            started_at=job['started_at'],
            completed_at=job['completed_at'],
            error_message=job['error_message']
        )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_extraction_error(e, "get_extraction_job_status", job_id=job_id)

@router.get("/projects/{project_id}/jobs", response_model=List[ExtractionJobStatus])
async def get_project_extraction_jobs(
    project_id: str,
    repos=Depends(get_repositories)
):
    """Get all extraction jobs for a project."""
    try:
        # Validate project_id - accept UUID strings
        if not project_id or not isinstance(project_id, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid project_id: must be a valid string identifier"
            )

        # Get all jobs for the project
        try:
            jobs_data = repos['extraction_job'].get_jobs_by_project(project_id)
        except Exception as e:
            raise handle_extraction_error(e, "get_project_jobs", project_id=project_id)

        jobs = []
        for job_data in jobs_data:
            # Get result metadata if job is completed
            progress = None
            if job_data['status'] == 'completed':
                try:
                    metadata = repos['hybrid_manager'].get_job_metadata(job_data['id'], storage_preference="hybrid")
                    if metadata:
                        progress = {
                            'total_results': metadata['total_results'],
                            'avg_confidence': metadata['avg_confidence'],
                            'categories_found': metadata['categories']
                        }
                except Exception as e:
                    api_logger.log_warning(
                        f"Failed to get result metadata for job {job_data['id']}: {e}",
                        "get_project_jobs",
                        context={"job_id": job_data['id'], "project_id": project_id}
                    )
                    # Fallback to database
                    try:
                        metadata = repos['extraction_result'].get_result_metadata(job_data['id'])
                        progress = {
                            'total_results': metadata['total_results'],
                            'avg_confidence': metadata['avg_confidence'],
                            'categories_found': metadata['categories']
                        }
                    except Exception:
                        pass

            # Convert string timestamps to datetime objects
            def parse_datetime(dt_str):
                if dt_str and isinstance(dt_str, str):
                    try:
                        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    except ValueError:
                        return None
                return dt_str

            # Ensure created_at is always present (required field)
            created_at = parse_datetime(job_data['created_at'])
            if created_at is None:
                created_at = datetime.utcnow()  # Fallback for invalid data

            # Extract taxonomy_id from parameters
            taxonomy_id = None
            if job_data.get('parameters'):
                try:
                    params = json.loads(job_data['parameters']) if isinstance(job_data['parameters'], str) else job_data['parameters']
                    taxonomy_id = params.get('taxonomy_id')
                except (json.JSONDecodeError, TypeError):
                    pass

            # Debug: Print the data being passed to the model
            job_status_data = {
                "job_id": job_data['id'],
                "taxonomy_id": taxonomy_id,
                "status": job_data['status'],
                "progress_percentage": float(job_data['progress'] or 0.0),
                "progress": progress,
                "created_at": created_at,
                "updated_at": parse_datetime(job_data['updated_at']),
                "started_at": parse_datetime(job_data['started_at']),
                "completed_at": parse_datetime(job_data['completed_at']),
                "error_message": job_data['error_message']
            }
            logger.debug(f"job_status_data: {job_status_data}")

            jobs.append(ExtractionJobStatus(**job_status_data))

        return jobs

    except HTTPException:
        raise
    except Exception as e:
        raise handle_extraction_error(e, "get_project_extraction_jobs", project_id=project_id)

@router.get("/{job_id}/results", response_model=ExtractionResultsResponse)
async def get_extraction_results(
    job_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    category_filter: Optional[str] = None,
    repos=Depends(get_repositories)
):
    """Get paginated results for an extraction job."""
    try:
        # Check if job exists and is completed
        job = repos['extraction_job'].get_job_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Extraction job with ID {job_id} not found")

        if job['status'] not in ['completed', 'processing']:
            raise HTTPException(
                status_code=400,
                detail=f"Job is not ready for results. Current status: {job['status']}"
            )

        # Get project ID from job
        project_id = job['project_id']

        # Get extraction results using hybrid storage manager
        try:
            paginated_results, has_more, total_results = repos['hybrid_manager'].retrieve_results(
                job_id=job_id,
                page=page,
                page_size=page_size,
                min_confidence=min_confidence,
                category_filter=category_filter,
                storage_preference="hybrid"  # Try filesystem first, fallback to database
            )

            # Convert to API response format
            formatted_results = []
            for result in paginated_results:
                formatted_results.append(ExtractionResultItem(
                    chunk_id=result.get('chunk_id', ''),
                    chunk_text=result.get('chunk_text'),
                    classifications=result.get('classifications', {}),
                    confidence_score=result.get('confidence_score', 0.0),
                    categories_matched=result.get('categories_matched', []),
                    metadata=result.get('metadata', {})
                ))
            paginated_results = formatted_results

        except Exception as e:
            api_logger.log_error(
                e,
                "hybrid_retrieval_failed",
                context={"job_id": job_id, "page": page, "page_size": page_size}
            )
            # Fallback to database-only retrieval
            results_data, has_more = repos['extraction_result'].get_results_by_job(
                job_id=job_id,
                page=page,
                page_size=page_size,
                min_confidence=min_confidence,
                category_filter=category_filter
            )

            paginated_results = []
            for row in results_data:
                try:
                    extracted_data = json.loads(row['extracted_data']) if isinstance(row['extracted_data'], str) else row['extracted_data']
                    paginated_results.append(ExtractionResultItem(
                        chunk_id=row['chunk_id'],
                        chunk_text=extracted_data.get('chunk_text', ''),
                        classifications=extracted_data.get('classifications', {}),
                        confidence_score=row['confidence'],
                        categories_matched=json.loads(row['categories']) if isinstance(row['categories'], str) else row['categories'],
                        metadata=extracted_data.get('metadata', {})
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    api_logger.log_warning(
                        f"Failed to parse result data for job {job_id}: {e}",
                        "get_extraction_results_fallback",
                        context={"job_id": job_id, "result_id": row['id']}
                    )
                    continue

            # Calculate pagination
            total_results = repos['extraction_result'].get_result_metadata(job_id).get('total_results', 0)

        # Build filters info
        filters = {}
        if min_confidence is not None:
            filters['min_confidence'] = min_confidence
        if category_filter:
            filters['category_filter'] = category_filter

        return ExtractionResultsResponse(
            job_id=job_id,
            results=paginated_results,
            total_results=total_results,
            page=page,
            page_size=page_size,
            has_more=has_more,
            filters=filters if filters else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get extraction results: {str(e)}")

@router.delete("/{job_id}", response_model=ExtractionJobCancellation)
async def cancel_extraction_job(
    job_id: str,
    repos=Depends(get_repositories)
):
    """Cancel an extraction job if it's still pending or processing."""
    try:
        job = repos['extraction_job'].get_job_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Extraction job with ID {job_id} not found")

        if job['status'] in ['completed', 'failed']:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job with status: {job['status']}"
            )

        # Update job status to cancelled
        repos['extraction_job'].update_job_status(job_id, "cancelled", "Job cancelled by user")

        # Clean up any partial results using hybrid storage
        try:
            repos['hybrid_manager'].delete_job_results(job_id, storage_preference="hybrid")
        except Exception as e:
            api_logger.log_warning(
                f"Failed to delete results for cancelled job {job_id}: {e}",
                "cancel_job_cleanup",
                context={"job_id": job_id}
            )
            # Fallback to database cleanup
            repos['extraction_result'].delete_results_by_job(job_id)

        return ExtractionJobCancellation(
            job_id=job_id,
            status="cancelled",
            message="Extraction job cancelled successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel extraction job: {str(e)}")

@router.delete("/{job_id}/delete", response_model=dict)
async def delete_extraction_job(
    job_id: str,
    repos=Depends(get_repositories)
):
    """Permanently delete an extraction job and all associated data."""
    try:
        # Validate job_id format
        import re
        if isinstance(job_id, str):
            if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', job_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid job_id: must be a valid UUID string"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid job_id: must be a UUID string"
            )

        # Check if job exists
        job = repos['extraction_job'].get_job_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Extraction job with ID {job_id} not found")

        # Prevent deletion of running jobs
        if job['status'] in ['pending', 'processing']:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot delete job with status: {job['status']}. Cancel the job first."
            )

        api_logger.log_operation_start(
            "delete_extraction_job",
            context={"job_id": job_id, "job_status": job['status']}
        )

        # Delete from filesystem first
        try:
            filesystem_success = repos['filesystem_manager'].delete_job_results(job_id)
            api_logger.log_warning(
                f"Filesystem deletion result for job {job_id}: {filesystem_success}",
                "filesystem_deletion",
                context={"job_id": job_id}
            )
        except Exception as e:
            api_logger.log_error(
                e,
                "filesystem_deletion_failed",
                context={"job_id": job_id}
            )
            # Continue with database deletion even if filesystem fails

        # Delete from database using a single transaction
        db = get_db()
        cursor = db.cursor()
        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")

            # First check how many results exist for this job
            cursor.execute("SELECT COUNT(*) FROM extraction_results WHERE job_id = ?", (job_id,))
            results_count_before = cursor.fetchone()[0]
            api_logger.log_warning(
                f"Found {results_count_before} extraction results for job {job_id} before deletion",
                "database_results_check",
                context={"job_id": job_id}
            )

            # Delete extraction results first
            cursor.execute("DELETE FROM extraction_results WHERE job_id = ?", (job_id,))
            results_deleted = cursor.rowcount
            api_logger.log_warning(
                f"Deleted {results_deleted} extraction results for job {job_id}",
                "database_results_deletion",
                context={"job_id": job_id, "results_deleted": results_deleted}
            )

            # Then delete the job record (CASCADE will handle any remaining results)
            cursor.execute("DELETE FROM extraction_jobs WHERE id = ?", (job_id,))
            job_deleted = cursor.rowcount
            api_logger.log_warning(
                f"Deleted {job_deleted} extraction job record for {job_id}",
                "database_job_deletion",
                context={"job_id": job_id, "job_deleted": job_deleted}
            )

            # Commit transaction
            db.commit()
            api_logger.log_operation_complete(
                "delete_extraction_job",
                duration=0.0,
                context={"job_id": job_id, "results_deleted": results_deleted, "job_deleted": job_deleted}
            )

        except Exception as e:
            # Rollback on any error
            db.rollback()
            api_logger.log_error(
                e,
                "database_deletion_failed",
                context={"job_id": job_id}
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete extraction job from database: {str(e)}"
            )

        api_logger.log_operation_complete(
            "delete_extraction_job",
            duration=0.0,  # Placeholder duration
            context={"job_id": job_id}
        )

        return {
            "job_id": job_id,
            "status": "deleted",
            "message": "Extraction job and all associated data deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.log_error(
            e,
            "delete_extraction_job_unexpected_error",
            context={"job_id": job_id}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting the extraction job"
        )

@router.post("/{job_id}/restart", response_model=ExtractionJobRestart)
async def restart_extraction_job(
    job_id: str,
    http_request: Request,
    repos=Depends(get_repositories)
):
    """Restart a failed or cancelled extraction job."""
    try:
        job = repos['extraction_job'].get_job_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Extraction job with ID {job_id} not found")

        if job['status'] not in ['failed', 'cancelled']:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot restart job with status: {job['status']}. Only failed or cancelled jobs can be restarted."
            )

        # Get job parameters for restarting
        parameters = job['parameters']

        # Submit job to the enhanced queue
        try:
            # Get job queue manager from app state
            job_queue_manager = get_job_queue_manager(http_request)
            if job_queue_manager is None:
                raise ValueError("Job queue manager not initialized")

            job_params = {
                "operation": "extraction",
                "taxonomy_id": parameters.get('taxonomy_id'),
                "selected_categories": parameters.get('selected_categories', []),
                "parameters": parameters.get('parameters', {}),
                "initial_classifier": parameters.get('initial_classifier', 'grok'),
                "enable_validation_stage": parameters.get('enable_validation_stage', False),
                "validation_classifier": parameters.get('validation_classifier'),
                "extraction_type": parameters.get('extraction_type', 'ner'),
                "extraction_mode": parameters.get('extraction_mode', 'contextual'),
            }
            job_id_str = job_queue_manager.submit_job(
                job_type=JobType.EXTRACTION,
                parameters=job_params
            )
        except Exception as e:
            raise handle_extraction_error(e, "restart_extraction_job", job_id=job_id)

        return ExtractionJobRestart(
            job_id=job_id,
            status="pending",
            message="Extraction job restarted and queued for processing"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart extraction job: {str(e)}")

from ...features.extraction.storage_monitor import (
    get_metrics,
    get_alert_manager,
    get_monitor,
    initialize_monitoring,
    StorageHealthMonitor
)

# Initialize monitoring for filesystem storage
def get_monitor_instance(repos=Depends(get_repositories)):
    """Get or initialize the storage monitor instance."""
    monitor = get_monitor()
    if monitor is None:
        monitor = initialize_monitoring(repos['filesystem_manager'])
    return monitor

@router.get("/storage/health")
async def get_storage_health(
    job_id: Optional[int] = None,
    repos=Depends(get_repositories),
    monitor=Depends(get_monitor_instance)
):
    """Get storage health status and integrity check results."""
    try:
        # Perform integrity check
        integrity_report = repos['filesystem_manager'].check_file_integrity(job_id)

        # Get health status from monitor
        health_status = monitor.get_health_status()

        return {
            'status': health_status['status'],
            'issues': health_status['issues'],
            'integrity_check': integrity_report,
            'last_health_check': health_status.get('last_check'),
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage health: {str(e)}")

@router.get("/storage/stats")
async def get_storage_statistics(repos=Depends(get_repositories)):
    """Get storage usage statistics."""
    try:
        # Calculate storage statistics
        base_path = repos['filesystem_manager'].file_manager.base_path

        total_files = 0
        total_size = 0
        projects_count = 0

        if base_path.exists():
            for project_dir in base_path.iterdir():
                if project_dir.is_dir() and project_dir.name.isdigit():
                    projects_count += 1
                    for file_path in project_dir.glob("*.json"):
                        total_files += 1
                        total_size += file_path.stat().st_size

        return {
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'projects_count': projects_count,
            'avg_file_size_kb': (total_size / max(total_files, 1)) / 1024,
            'storage_path': str(base_path),
            'last_updated': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage statistics: {str(e)}")

@router.get("/storage/metrics")
async def get_storage_metrics():
    """Get storage performance metrics."""
    try:
        metrics = get_metrics().get_metrics()
        return {
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage metrics: {str(e)}")

@router.get("/storage/alerts")
async def get_storage_alerts(
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get storage alerts with optional filtering."""
    try:
        alert_manager = get_alert_manager()
        alerts = alert_manager.get_alerts(
            severity=severity,
            acknowledged=acknowledged,
            limit=limit
        )
        return {
            'alerts': alerts,
            'total_count': len(alerts),
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage alerts: {str(e)}")

@router.post("/storage/alerts/{alert_id}/acknowledge")
async def acknowledge_storage_alert(alert_id: str):
    """Acknowledge a storage alert."""
    try:
        alert_manager = get_alert_manager()
        alert_manager.acknowledge_alert(alert_id)
        return {
            'message': f'Alert {alert_id} acknowledged',
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.post("/storage/maintenance/integrity-check")
async def run_integrity_check(
    job_id: Optional[int] = None,
    repos=Depends(get_repositories)
):
    """Run a manual integrity check on storage files."""
    try:
        integrity_report = repos['filesystem_manager'].check_file_integrity(job_id)

        # Trigger alerts for any issues found
        alert_manager = get_alert_manager()
        if integrity_report['corrupted_files'] > 0:
            alert_manager.trigger_alert(
                'maintenance',
                'warning',
                f"Integrity check found {integrity_report['corrupted_files']} corrupted files",
                {'corrupted_files': integrity_report['corrupted_files_list']}
            )

        return {
            'integrity_check': integrity_report,
            'message': 'Integrity check completed',
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run integrity check: {str(e)}")

@router.post("/storage/maintenance/cleanup")
async def cleanup_storage(
    days_old: int = Query(30, ge=1, le=365),
    repos=Depends(get_repositories)
):
    """Clean up old storage files and directories."""
    try:
        # This is a simplified cleanup - in production you'd want more sophisticated logic
        base_path = repos['filesystem_manager'].file_manager.base_path
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        cleaned_files = 0
        cleaned_projects = 0

        if base_path.exists():
            for project_dir in base_path.iterdir():
                if project_dir.is_dir() and project_dir.name.isdigit():
                    # Note: We don't delete project directories during cleanup
                    # because they might contain fresh files from other jobs.
                    # We only clean up old files within them.
                    try:
                        project_id = int(project_dir.name)
                        for file_path in project_dir.glob("*.json"):
                            stat = file_path.stat()
                            file_date = datetime.fromtimestamp(stat.st_mtime)
                            if file_date < cutoff_date:
                                file_path.unlink()
                                cleaned_files += 1
                    except ValueError:
                        continue

        return {
            'cleaned_files': cleaned_files,
            'days_old_threshold': days_old,
            'message': f'Cleaned up {cleaned_files} files in project directories',
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup storage: {str(e)}")
