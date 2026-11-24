"""Documents routes for the Compileo API."""

import os
import uuid
import json
import glob
from typing import List, Optional, Literal
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import ProjectRepository, DocumentRepository, DatasetJobRepository
from ..core.database import get_db
from ..core.dependencies import get_current_active_user
from ..models.user import User as UserSchema
from ...features.extraction.error_logging import api_logger
from ...features.chunk.engine import chunk_document
from ...features.chunk.schema import CharacterStrategy, LLMPromptStrategy, SchemaStrategy, DelimiterStrategy, TokenStrategy
from ...features.ingestion.processing.pdf_splitter import pre_split_pdf

# Create router
router = APIRouter()

import logging
logger = logging.getLogger(__name__)

# Pydantic models
class DocumentResponse(BaseModel):
    id: str
    project_id: str
    file_name: str
    source_file_path: Optional[str] = None
    created_at: datetime
    status: str = "uploaded"
    page_selection: Optional[str] = None
    # Parsing information from related tables
    parsed_documents: List[dict] = []
    parsed_files: List[dict] = []

class UploadResponse(BaseModel):
    job_id: str
    message: str
    files_count: int
    uploaded_files: List[DocumentResponse]

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    current_step: str
    estimated_completion: datetime
    processed_files: List[DocumentResponse]

class ProcessDocumentsRequest(BaseModel):
    project_id: str
    document_ids: List[str]
    parser: str = Field(default="grok", description="Parser to use for document processing (grok, gemini)")
    chunker: str = Field(default="grok", description="Legacy chunker field for backward compatibility")
    model: Optional[str] = Field(default=None, description="Specific model name to use (overrides settings)")
    chunk_size: int = Field(default=512, description="Default chunk size for token-based chunking")
    overlap: int = Field(default=50, description="Overlap between chunks for token-based chunking")
    chunk_strategy: Optional[Literal["token", "character", "semantic", "delimiter", "schema"]] = Field(
        default=None,
        description="Chunking strategy to use (token, character, semantic, delimiter, schema)"
    )
    semantic_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for semantic chunking"
    )
    schema_definition: Optional[str] = Field(
        default=None,
        description="JSON schema definition for schema-based chunking"
    )
    character_chunk_size: Optional[int] = Field(
        default=None,
        description="Chunk size for character-based chunking"
    )
    character_overlap: Optional[int] = Field(
        default=None,
        description="Overlap for character-based chunking"
    )
    num_ctx: Optional[int] = Field(
        default=None,
        description="Context window size for Ollama models (overrides default setting)"
    )
    system_instruction: Optional[str] = Field(
        default=None,
        description="System-level instructions to guide the model's behavior, especially for Gemini"
    )
    skip_parsing: bool = Field(default=False, description="Skip parsing if documents are already parsed")
    sliding_window: bool = Field(default=True, description="Enable sliding window chunking for parsed documents (default: True)")
    pages_per_split: int = Field(default=50, description="Number of pages per split for PDF documents")
    overlap_pages: int = Field(default=0, description="Number of overlapping pages between splits")
    page_selection: Optional[str] = Field(
        default=None,
        description="Page selection expression for PDF documents (e.g., '1', '1-20', '1, 1-20, 30-120'). Null means all pages."
    )

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int

class SplitPDFRequest(BaseModel):
    pdf_path: str
    pages_per_split: int = Field(default=200, description="Number of pages per split chunk")
    overlap_pages: int = Field(default=1, description="Number of overlapping pages between splits")

class SplitPDFResponse(BaseModel):
    split_files: List[str]
    message: str
    total_splits: int
    manifest_path: Optional[str] = None

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    project_id: str = Form(...),
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Upload multiple documents to a project for current user."""
    try:
        project_repo = ProjectRepository(current_user.id)
        document_repo = DocumentRepository(current_user.id)

        # Validate project exists and belongs to user
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?", (project_id, current_user.id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

        job_id = str(uuid.uuid4())
        uploaded_files = []

        # Create user-scoped uploads directory
        upload_dir = f"storage/{current_user.id}/uploads/{project_id}"
        os.makedirs(upload_dir, exist_ok=True)

        for file in files:
            # Validate file name exists
            if file.filename is None:
                raise HTTPException(status_code=400, detail="File has no name")

            # Validate file type
            allowed_extensions = {
                '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
                '.txt', '.md', '.csv', '.json', '.xml'
            }
            file_ext = os.path.splitext(file.filename)[1].lower()

            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
                )

            # Generate unique filename
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = os.path.join(upload_dir, unique_filename)

            # Save file
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Create document record
            document_id = document_repo.create_document(project_id, file.filename)
            document_repo.update_document_path(document_id, file_path)

            uploaded_files.append(DocumentResponse(
                id=document_id,
                project_id=project_id,
                file_name=file.filename,
                source_file_path=file_path,
                created_at=datetime.utcnow(),
                parsed_documents=[],
                parsed_files=[]
            ))

        return UploadResponse(
            job_id=job_id,
            message="Documents uploaded successfully",
            files_count=len(uploaded_files),
            uploaded_files=uploaded_files
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload documents: {str(e)}")

@router.post("/process")
async def process_documents(
    request: ProcessDocumentsRequest,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Process uploaded documents by submitting jobs to the RQ worker."""
    logger.debug(f"SERVER DEBUG: process_documents route called with request: {request}")
    logger.debug(f"SERVER DEBUG: project_id={request.project_id}, document_ids={request.document_ids}, parser={request.parser}, chunker={request.chunker}")
    try:
        project_repo = ProjectRepository(current_user.id)
        document_repo = DocumentRepository(current_user.id)

        # Validate project exists and belongs to user
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?", (request.project_id, current_user.id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")

        # Validate documents exist and belong to project and user
        logger.debug(f"SERVER DEBUG: Validating {len(request.document_ids)} documents for project {request.project_id}")
        for doc_id in request.document_ids:
            cursor.execute("SELECT id FROM documents WHERE id = ? AND project_id = ? AND owner_id = ?", (doc_id, request.project_id, current_user.id))
            doc_exists = cursor.fetchone()
            logger.debug(f"SERVER DEBUG: Document {doc_id} in project {request.project_id}: {'FOUND' if doc_exists else 'NOT FOUND'}")
            if not doc_exists:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {doc_id} not found or doesn't belong to project {request.project_id}"
                )

        # Submit job to RQ worker instead of processing directly
        from ...features.gui.services.async_job_submission_service import submit_document_processing_job

        # Determine operation based on skip_parsing flag
        from ...core.settings import BackendSettings
        
        if request.skip_parsing:
            # Skip parsing, go straight to chunking
            operation = "chunk_documents"
            # Auto-detect schema strategy if schema_definition is provided
            if request.schema_definition:
                effective_chunk_strategy = "schema"
                logger.debug(f"DEBUG_API: Auto-detected schema strategy from schema_definition")
            else:
                effective_chunk_strategy = request.chunk_strategy or "character"
            if effective_chunk_strategy.strip().lower() == "schema":
                from ...features.chunk.schema import parse_chunking_schema

                try:
                    parse_chunking_schema(request.schema_definition or "")
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
            logger.debug(f"DEBUG_API: effective_chunk_strategy = {repr(effective_chunk_strategy)}")

            # Retrieve parsed files data for multi-file documents from new schema
            manifest_data = None
            if request.document_ids:
                import sys
                logger.debug(f"DEBUG_API: Querying parsed files for document {request.document_ids[0]}")
                cursor.execute("""
                    SELECT pf.file_path
                    FROM parsed_documents pd
                    JOIN parsed_files pf ON pd.id = pf.parsed_document_id
                    WHERE pd.document_id = ?
                    ORDER BY pf.page_number
                """, (request.document_ids[0],))
                parsed_files_rows = cursor.fetchall()
                logger.debug(f"DEBUG_API: Query returned {len(parsed_files_rows)} rows: {parsed_files_rows}")
                if parsed_files_rows:
                    manifest_data = {
                        "parsed_files": [row[0] for row in parsed_files_rows],
                        "document_id": request.document_ids[0]
                    }
                    logger.debug(f"DEBUG_API: Set manifest_data with {len(parsed_files_rows)} files: {manifest_data}")
                else:
                    logger.debug(f"DEBUG_API: No parsed files found for document {request.document_ids[0]}")

            # Get the actual model name - prefer request model over settings
            chunking_provider = request.chunker or "gemini"
            # Check if model was explicitly passed in the request
            request_model = getattr(request, 'model', None)
            if request_model:
                chunking_model_name = request_model
            else:
                # Fall back to user settings if no model in request
                settings = BackendSettings(owner_id=current_user.id)
                if chunking_provider == "ollama":
                    chunking_model_name = settings.get_chunking_ollama_model()
                elif chunking_provider == "gemini":
                    chunking_model_name = settings.get_chunking_gemini_model()
                elif chunking_provider == "grok":
                    chunking_model_name = settings.get_chunking_grok_model()
                elif chunking_provider == "openai":
                    chunking_model_name = settings.get_chunking_openai_model()
                else:
                    chunking_model_name = chunking_provider

            job_parameters = {
                "operation": operation,
                "owner_id": current_user.id,
                "project_id": request.project_id,
                "document_ids": request.document_ids,
                "chunker": chunking_provider,
                "model": chunking_model_name,
                "chunk_strategy": effective_chunk_strategy,
                "skip_parsing": True,
                "manifest_data": manifest_data,
                "sliding_window": request.sliding_window,
                "system_instruction": request.system_instruction
            }

            # Add strategy-specific parameters
            if effective_chunk_strategy == "character":
                job_parameters.update({
                    "chunk_size": request.character_chunk_size or request.chunk_size,
                    "overlap": request.character_overlap or request.overlap,
                    "num_ctx": request.num_ctx
                })
            elif effective_chunk_strategy == "token":
                job_parameters.update({
                    "chunk_size": request.chunk_size,
                    "overlap": request.overlap,
                    "num_ctx": request.num_ctx
                })
            elif effective_chunk_strategy == "semantic":
                job_parameters.update({
                    "semantic_prompt": request.semantic_prompt,
                    "num_ctx": request.num_ctx
                })
                logger.debug(f"DEBUG_API: Final job parameters model: {repr(job_parameters.get('model'))}")
            elif effective_chunk_strategy == "schema":
                job_parameters.update({
                    "schema_definition": request.schema_definition,
                    "num_ctx": request.num_ctx
                })
            elif effective_chunk_strategy == "delimiter":
                job_parameters.update({
                    "delimiters": ["\n\n", "\n"],  # Default delimiters
                    "num_ctx": request.num_ctx
                })
        else:
            # Parse documents first - submit each document as its own job
            operation = "parse_documents"
            
            # Get specific model for parser - prefer request model over settings
            request_model = getattr(request, 'model', None)
            if request_model:
                parsing_model_name = request_model
            else:
                # Fall back to user settings if no model in request
                settings = BackendSettings(owner_id=current_user.id)
                if request.parser == "gemini":
                    parsing_model_name = settings.get_parsing_gemini_model()
                elif request.parser == "grok":
                    parsing_model_name = settings.get_parsing_grok_model()
                elif request.parser == "ollama":
                    parsing_model_name = settings.get_parsing_ollama_model()
                elif request.parser == "openai":
                    parsing_model_name = settings.get_parsing_openai_model()
                else:
                    parsing_model_name = request.parser

            # Submit each document as a separate job for parallel processing
            from ...features.jobhandle.models import JobPriority
            from ...features.gui.services.async_job_submission_service import submit_document_processing_job
            
            job_ids = []
            for doc_id in request.document_ids:
                # Save page selection to document record if provided
                if request.page_selection is not None:
                    document_repo.update_page_selection(doc_id, request.page_selection)

                job_parameters = {
                    "operation": operation,
                    "owner_id": current_user.id,
                    "project_id": request.project_id,
                    "document_ids": [doc_id],  # Single document per job
                    "parser": request.parser,
                    "model": parsing_model_name,
                    "pages_per_split": request.pages_per_split,
                    "overlap_pages": request.overlap_pages,
                    "page_selection": request.page_selection
                }

                job_id = submit_document_processing_job(
                    parameters=job_parameters,
                    priority=JobPriority.HIGH,
                    user_id=current_user.id
                )
                job_ids.append(job_id)
                logger.info(f"Submitted parse job {job_id} for document {doc_id}")

            result = {
                "job_id": job_ids[0] if len(job_ids) == 1 else job_ids,
                "job_ids": job_ids,
                "message": f"Submitted {len(job_ids)} document parsing jobs. Track progress with job IDs: {job_ids}",
                "status": "submitted",
                "estimated_duration": "Processing in background",
                "debug_info": {
                    "total_requested": len(request.document_ids),
                    "jobs_submitted": len(job_ids),
                    "project_id": request.project_id,
                    "parser": request.parser,
                    "chunk_strategy": request.chunk_strategy,
                    "chunker": request.chunker
                }
            }
            logger.debug(f"DEBUG: Jobs submitted successfully: {result}")
            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.debug(f"DEBUG: Exception in process_documents: {e}")
        import traceback
        logger.debug(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to submit document processing job: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Delete a document and all its associated files and chunks."""
    cursor = None
    try:
        document_repo = DocumentRepository(current_user.id)
        cleanup_repo = DatasetJobRepository(current_user.id)

        # Get document info
        cursor = db.cursor()
        cursor.execute("""
            SELECT project_id, file_name, source_file_path FROM documents
            WHERE id = ? AND owner_id = ?
        """, (document_id, current_user.id))

        doc_record = cursor.fetchone()
        if not doc_record:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")

        project_id, file_name, source_file_path = doc_record

        # Get all parsed files for this document to clean up filesystem BEFORE cleanup
        cursor.execute("""
            SELECT pf.file_path
            FROM parsed_files pf
            JOIN parsed_documents pd ON pf.parsed_document_id = pd.id
            WHERE pd.document_id = ?
        """, (document_id,))
        parsed_file_rows = cursor.fetchall()

        # Get all chunk files for this document BEFORE cleanup
        cursor.execute("SELECT file_path FROM chunks WHERE document_id = ?", (document_id,))
        chunk_rows = cursor.fetchall()

        # Combine all file paths that need cleanup
        output_rows = parsed_file_rows + chunk_rows

        # Perform comprehensive database cleanup for this document
        # Manual cleanup since we removed the method from repo for now
        cursor.execute("DELETE FROM extraction_results WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)", (document_id,))
        cursor.execute("DELETE FROM chunks WHERE document_id = ? AND owner_id = ?", (document_id, current_user.id))
        cursor.execute("DELETE FROM parsed_files WHERE parsed_document_id IN (SELECT id FROM parsed_documents WHERE document_id = ?)", (document_id,))
        cursor.execute("DELETE FROM parsed_documents WHERE document_id = ?", (document_id,))
        db.commit()

        # Delete document record from database (do this after cleanup to avoid FK issues)
        document_repo.delete_document(document_id)

        # Clean up filesystem files (based on project deletion logic but for single document)
        import os
        import shutil

        # Get base directory for absolute path calculations
        base_dir = os.getcwd()

        try:
            # Clean up the original document file (database-mediated)
            if source_file_path:
                # Convert relative paths to absolute paths
                if not os.path.isabs(source_file_path):
                    source_file_path = os.path.join(base_dir, source_file_path)

                try:
                    # Remove the file
                    os.remove(source_file_path)
                    logger.debug(f"Successfully removed document file: {source_file_path}")
                    # Try to remove the parent directory if it's a project-specific directory and empty
                    parent_dir = os.path.dirname(source_file_path)
                    if parent_dir != os.path.join(base_dir, "uploads") and os.path.exists(parent_dir):
                        # Check if directory is empty
                        if not os.listdir(parent_dir):
                            os.rmdir(parent_dir)
                            logger.debug(f"Successfully removed empty directory: {parent_dir}")
                except FileNotFoundError:
                    logger.debug(f"Document file already removed: {source_file_path}")
                except Exception as e:
                    # Log but don't fail the operation
                    logger.warning(f"Warning: Failed to remove document file {source_file_path}: {e}")

            # Clean up processed output files (chunks, parsed files, etc.) (database-mediated)
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

            # Clean up parsed files (from the database query above, output_rows contains all file paths)
            # The parsed files are already included in output_rows from the database query
            # No need for separate manifest processing since we get file paths directly from DB

            # Clean up split PDF files that may have been created during preprocessing
            if source_file_path and file_name:
                # Get the original source directory (before conversion to absolute)
                # Use the source_file_path that was already retrieved before deletion
                source_dir = os.path.dirname(source_file_path) if os.path.isabs(source_file_path) else os.path.dirname(os.path.join(base_dir, source_file_path))
                base_name = os.path.splitext(file_name)[0]

                # Look for split PDF manifest
                split_manifest_path = os.path.join(source_dir, f"{base_name}_manifest.json")
                if os.path.exists(split_manifest_path):
                    try:
                        with open(split_manifest_path, 'r', encoding='utf-8') as f:
                            split_manifest_data = json.load(f)

                        # Delete all split PDF files listed in the manifest
                        splits_info = split_manifest_data.get("splits", [])
                        for split_info in splits_info:
                            split_file_path = split_info.get("file_path")
                            if split_file_path and os.path.exists(split_file_path):
                                try:
                                    os.remove(split_file_path)
                                    logger.debug(f"Successfully removed split PDF file: {split_file_path}")
                                except Exception as e:
                                    logger.warning(f"Warning: Failed to remove split PDF file {split_file_path}: {e}")

                        # Delete the split manifest file itself
                        os.remove(split_manifest_path)
                        logger.debug(f"Successfully removed split PDF manifest: {split_manifest_path}")

                    except Exception as e:
                        logger.warning(f"Warning: Failed to process split PDF manifest {split_manifest_path}: {e}")

                # Also look for any remaining split files that might not be in manifest
                try:
                    split_pattern = os.path.join(source_dir, f"{base_name}_chunk_*.pdf")
                    split_files = glob.glob(split_pattern)
                    for split_file in split_files:
                        try:
                            os.remove(split_file)
                            logger.debug(f"Successfully removed orphaned split PDF file: {split_file}")
                        except Exception as e:
                            logger.warning(f"Warning: Failed to remove orphaned split PDF file {split_file}: {e}")
                except Exception as e:
                    logger.warning(f"Warning: Failed to cleanup orphaned split PDF files: {e}")

            # Create a new cursor for post-commit operations
            cleanup_cursor = db.cursor()

            # Clean up project-specific directories if they become empty (mediated by DB metadata)
            cleanup_cursor.execute("SELECT COUNT(*) FROM documents WHERE project_id = ?", (project_id,))
            remaining_docs = cleanup_cursor.fetchone()[0]

            if remaining_docs == 0:  # No documents left in project, clean up all project directories
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
                        shutil.rmtree(full_dir_path)
                        logger.debug(f"Successfully removed empty project directory: {full_dir_path}")
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        logger.warning(f"Warning: Failed to remove project directory {full_dir_path}: {e}")
            else:
                # Clean up only document-specific chunks if they exist
                doc_chunks_dir = os.path.join(base_dir, f"storage/{current_user.id}/chunks/{project_id}/{document_id}")
                try:
                    shutil.rmtree(doc_chunks_dir)
                    logger.debug(f"Successfully removed document chunks directory: {doc_chunks_dir}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.warning(f"Warning: Failed to remove document chunks directory {doc_chunks_dir}: {e}")
        except Exception as e:
            # Log filesystem cleanup errors but don't fail the operation
            logger.warning(f"Warning: Filesystem cleanup failed: {e}")
            import traceback
            logger.warning(f"Filesystem cleanup traceback: {traceback.format_exc()}")

        return {"message": f"Document '{file_name}' and all associated files deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"ERROR: Failed to delete document: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    project_id: Optional[str] = None,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """List documents for current user, optionally filtered by project."""
    try:
        cursor = db.cursor()

        if project_id:
            # Validate project exists and belongs to user
            cursor.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?", (project_id, current_user.id))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

            # Get documents for specific project and user
            cursor.execute("""
                SELECT id, project_id, file_name, source_file_path, created_at, status, page_selection
                FROM documents
                WHERE project_id = ? AND owner_id = ?
                ORDER BY created_at DESC
            """, (project_id, current_user.id))
        else:
            # Get all documents for user
            cursor.execute("""
                SELECT id, project_id, file_name, source_file_path, created_at, status, page_selection
                FROM documents
                WHERE owner_id = ?
                ORDER BY created_at DESC
            """, (current_user.id,))

        documents = []
        for row in cursor.fetchall():
            doc_id, proj_id, file_name, source_path, created_at, status, page_selection = row

            # Get parsing information for this document
            cursor.execute("""
                SELECT id, parser_used, parse_config, total_pages, parsing_time, created_at
                FROM parsed_documents
                WHERE document_id = ?
                ORDER BY created_at DESC
            """, (doc_id,))
            parsed_docs = cursor.fetchall()

            parsed_documents = []
            parsed_files = []

            for parsed_doc in parsed_docs:
                pd_id, parser_used, parse_config, total_pages, parsing_time, pd_created_at = parsed_doc
                parsed_documents.append({
                    "id": pd_id,
                    "parser_used": parser_used,
                    "parse_config": parse_config,
                    "total_pages": total_pages,
                    "parsing_time": parsing_time,
                    "created_at": datetime.fromisoformat(pd_created_at) if isinstance(pd_created_at, str) else pd_created_at
                })

                # Get files for this parsed document
                cursor.execute("""
                    SELECT file_path, page_number, content_length, created_at
                    FROM parsed_files
                    WHERE parsed_document_id = ?
                    ORDER BY page_number
                """, (pd_id,))
                files = cursor.fetchall()
                for file_path, page_number, content_length, pf_created_at in files:
                    parsed_files.append({
                        "parsed_document_id": pd_id,
                        "file_path": file_path,
                        "page_number": page_number,
                        "content_length": content_length,
                        "created_at": datetime.fromisoformat(pf_created_at) if isinstance(pf_created_at, str) else pf_created_at
                    })

            documents.append(DocumentResponse(
                id=doc_id,
                project_id=proj_id,
                file_name=file_name,
                source_file_path=source_path,
                created_at=datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at,
                status=status,
                page_selection=page_selection,
                parsed_documents=parsed_documents,
                parsed_files=parsed_files
            ))

        return {"documents": documents, "total": len(documents)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get a specific document by ID for current user."""
    try:
        cursor = db.cursor()

        # Get document info
        cursor.execute("""
            SELECT id, project_id, file_name, source_file_path, created_at, status, page_selection
            FROM documents
            WHERE id = ? AND owner_id = ?
        """, (document_id, current_user.id))

        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")

        doc_id, proj_id, file_name, source_path, created_at, status, page_selection = row

        # Get parsing information for this document
        cursor.execute("""
            SELECT id, parser_used, parse_config, total_pages, parsing_time, created_at
            FROM parsed_documents
            WHERE document_id = ?
            ORDER BY created_at DESC
        """, (doc_id,))
        parsed_docs = cursor.fetchall()

        parsed_documents = []
        parsed_files = []

        for parsed_doc in parsed_docs:
            pd_id, parser_used, parse_config, total_pages, parsing_time, pd_created_at = parsed_doc
            parsed_documents.append({
                "id": pd_id,
                "parser_used": parser_used,
                "parse_config": parse_config,
                "total_pages": total_pages,
                "parsing_time": parsing_time,
                "created_at": datetime.fromisoformat(pd_created_at) if isinstance(pd_created_at, str) else pd_created_at
            })

            # Get files for this parsed document
            cursor.execute("""
                SELECT file_path, page_number, content_length, created_at
                FROM parsed_files
                WHERE parsed_document_id = ?
                ORDER BY page_number
            """, (pd_id,))
            files = cursor.fetchall()
            for file_path, page_number, content_length, pf_created_at in files:
                parsed_files.append({
                    "parsed_document_id": pd_id,
                    "file_path": file_path,
                    "page_number": page_number,
                    "content_length": content_length,
                    "created_at": datetime.fromisoformat(pf_created_at) if isinstance(pf_created_at, str) else pf_created_at
                })

        return DocumentResponse(
            id=doc_id,
            project_id=proj_id,
            file_name=file_name,
            source_file_path=source_path,
            created_at=datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at,
            status=status,
            page_selection=page_selection,
            parsed_documents=parsed_documents,
            parsed_files=parsed_files
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@router.get("/upload/{job_id}/status", response_model=ProcessingStatus)
async def get_upload_status(job_id: str, db=Depends(get_db)):
    """Get status of document upload/processing job."""
    try:
        # TODO: Implement proper job status tracking
        # For now, return mock status with uploaded files
        # In a real implementation, this would track actual upload jobs
        document_repo = DocumentRepository(current_user.id)

        # Get all documents (since we don't have job tracking yet)
        # This is a temporary solution - proper job tracking should be implemented
        cursor = db.cursor()
        cursor.execute("""
            SELECT id, project_id, file_name, source_file_path, created_at
            FROM documents
            ORDER BY created_at DESC
            LIMIT 10
        """)

        processed_files = []
        for row in cursor.fetchall():
            doc_id, proj_id, file_name, source_path, created_at = row
            processed_files.append(DocumentResponse(
                id=doc_id,
                project_id=proj_id,
                file_name=file_name,
                source_file_path=source_path,
                created_at=created_at
            ))

        return ProcessingStatus(
            job_id=job_id,
            status="completed",
            progress=100,
            current_step="Upload complete",
            estimated_completion=datetime.utcnow(),
            processed_files=processed_files
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.get("/process/{job_id}/status", response_model=ProcessingStatus)
async def get_processing_status(job_id: str, db=Depends(get_db)):
    """Get status of document processing job."""
    try:
        # Since processing is synchronous, return completed status
        # TODO: Implement proper job tracking for async processing if needed
        return ProcessingStatus(
            job_id=job_id,
            status="completed",
            progress=100,
            current_step="Processing complete",
            estimated_completion=datetime.utcnow(),
            processed_files=[]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get processing status: {str(e)}")

@router.get("/{document_id}/content")
async def get_document_content(
    document_id: str,
    page: int = 1,
    page_size: int = 10000,
    parsed_file: Optional[str] = None,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get the parsed content of a document with pagination support for current user."""
    try:
        cursor = db.cursor()

        # Get document info
        cursor.execute("""
            SELECT project_id, file_name, source_file_path, status FROM documents
            WHERE id = ? AND owner_id = ?
        """, (document_id, current_user.id))

        doc_record = cursor.fetchone()
        if not doc_record:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")

        project_id, file_name, source_file_path, status = doc_record

        # Get parsed documents and files from new schema
        cursor.execute("""
            SELECT pf.file_path
            FROM parsed_documents pd
            JOIN parsed_files pf ON pd.id = pf.parsed_document_id
            WHERE pd.document_id = ? AND pd.owner_id = ?
            ORDER BY pf.page_number, pf.file_path
        """, (document_id, current_user.id))

        parsed_files_rows = cursor.fetchall()
        parsed_files_list = [row[0] for row in parsed_files_rows]

        if status != 'parsed':
            raise HTTPException(status_code=400, detail=f"Document {document_id} is not parsed yet (status: {status})")

        # Check if this document has multiple parsed files (from new schema)
        logger.debug(f"SERVER DEBUG: Checking document {document_id} - found {len(parsed_files_list)} parsed files from new schema, parsed_file: {parsed_file}")
        logger.debug(f"SERVER DEBUG: Current working directory: {os.getcwd()}")

        # If a specific parsed file is requested, return its FULL content (no API pagination)
        logger.debug(f"SERVER DEBUG: parsed_file: {parsed_file}, parsed_files_list length: {len(parsed_files_list)}")
        if parsed_file and parsed_files_list:
            logger.debug(f"SERVER DEBUG: Branch 1 - Requesting full content of parsed file: {parsed_file}")

            # Find the requested file in the manifest
            target_file = None
            for pf in parsed_files_list:
                if os.path.basename(pf) == parsed_file or pf == parsed_file:
                    target_file = pf
                    break

            if not target_file or not os.path.exists(target_file):
                raise HTTPException(status_code=404, detail=f"Requested parsed file not found: {parsed_file}")

            # Read the FULL content of the specific file
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    raw_content = f.read()

                # Check if this is a JSON file with structured content
                full_content = raw_content
                try:
                    json_data = json.loads(raw_content)
                    if isinstance(json_data, dict) and 'main_content' in json_data:
                        # Extract main_content from JSON structure
                        full_content = json_data['main_content']
                        logger.debug(f"SERVER DEBUG: Extracted main_content from JSON file {target_file}, length: {len(full_content)}")
                    else:
                        logger.debug(f"SERVER DEBUG: File {target_file} is not JSON-structured, using as plain text")
                except json.JSONDecodeError:
                    logger.debug(f"SERVER DEBUG: File {target_file} is not valid JSON, using as plain text")

                logger.debug(f"SERVER DEBUG: Read {len(full_content)} chars from {target_file}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read parsed file {target_file}: {str(e)}")

            # Return the complete file content - GUI will handle pagination
            return {
                "document_id": document_id,
                "file_name": file_name,
                "content": full_content,
                "total_length": len(full_content),
                "word_count": len(full_content.split()),
                "line_count": len(full_content.splitlines()),
                "current_file": os.path.basename(target_file),
                "parsed_file": parsed_file
            }

        # If no specific file requested but we have multiple files, return file list
        elif parsed_files_list:
            logger.debug(f"SERVER DEBUG: Returning list of {len(parsed_files_list)} parsed files")
            return {
                "document_id": document_id,
                "file_name": file_name,
                "parsed_files": [os.path.basename(pf) for pf in parsed_files_list],
                "total_files": len(parsed_files_list),
                "schema_info": "Using new parsed_documents/parsed_files schema"
            }

        # Single file or fallback behavior
        else:
            full_content = ""
            if parsed_files_list and len(parsed_files_list) == 1:
                single_file_path = parsed_files_list[0]
                logger.debug(f"SERVER DEBUG: Using single parsed file for document {document_id}: {single_file_path}")
                try:
                    with open(single_file_path, 'r', encoding='utf-8') as f:
                        raw_content = f.read()

                    # Check if this is a JSON file with structured content
                    full_content = raw_content
                    try:
                        json_data = json.loads(raw_content)
                        if isinstance(json_data, dict) and 'main_content' in json_data:
                            # Extract main_content from JSON structure
                            full_content = json_data['main_content']
                            logger.debug(f"SERVER DEBUG: Extracted main_content from JSON file {single_file_path}, length: {len(full_content)}")
                        else:
                            logger.debug(f"SERVER DEBUG: File {single_file_path} is not JSON-structured, using as plain text")
                    except json.JSONDecodeError:
                        logger.debug(f"SERVER DEBUG: File {single_file_path} is not valid JSON, using as plain text")

                    logger.debug(f"SERVER DEBUG: Read {len(full_content)} chars from single file {single_file_path}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to read parsed content: {str(e)}")
            else:
                logger.debug(f"SERVER DEBUG: No parsed content found for document {document_id} (found {len(parsed_files_list)} files)")
                raise HTTPException(status_code=404, detail=f"Parsed content not found for document {document_id}")

            # Calculate pagination for single file
            total_length = len(full_content)
            total_pages = (total_length + page_size - 1) // page_size  # Ceiling division

            # Validate page number
            if page < 1:
                page = 1
            if page > total_pages:
                page = total_pages

            # Extract page content
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_length)
            page_content = full_content[start_idx:end_idx]

            return {
                "document_id": document_id,
                "file_name": file_name,
                "content": page_content,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_length": total_length,
                "has_previous": page > 1,
                "has_next": page < total_pages,
                "word_count": len(full_content.split()),
                "line_count": len(full_content.splitlines())
            }

        if not full_content:
            raise HTTPException(status_code=404, detail=f"No content available for document {document_id}")

        # Calculate pagination
        total_length = len(full_content)
        total_pages = (total_length + page_size - 1) // page_size  # Ceiling division

        # Validate page number
        if page < 1:
            page = 1
        if page > total_pages:
            page = total_pages

        # Extract page content
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_length)
        page_content = full_content[start_idx:end_idx]

        return {
            "document_id": document_id,
            "file_name": file_name,
            "content": page_content,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "total_length": total_length,
            "has_previous": page > 1,
            "has_next": page < total_pages,
            "word_count": len(full_content.split()),
            "line_count": len(full_content.splitlines())
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document content: {str(e)}")

@router.post("/process/debug")
async def debug_process_documents(db=Depends(get_db)):
    """A debug endpoint to process a hardcoded document."""
    request = ProcessDocumentsRequest(
        project_id="1",
        document_ids=["1"],
        parser="grok",
        chunker="grok"
    )
    return await process_documents(request, db)

@router.post("/split-pdf", response_model=SplitPDFResponse)
async def split_pdf(request: SplitPDFRequest):
    """Split a large PDF into smaller chunks for processing."""
    try:
        # Validate PDF path exists
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {request.pdf_path}")

        # Validate PDF extension
        if not request.pdf_path.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Perform the split
        split_result = pre_split_pdf(
            file_path=request.pdf_path,
            pages_per_split=request.pages_per_split,
            overlap_pages=request.overlap_pages
        )

        split_files = split_result["split_files"]
        manifest_path = split_result["manifest_path"]

        return SplitPDFResponse(
            split_files=split_files,
            message=f"Successfully split PDF into {len(split_files)} files",
            total_splits=len(split_files),
            manifest_path=manifest_path
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF splitting failed: {str(e)}")
