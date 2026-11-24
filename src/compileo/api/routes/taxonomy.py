"""Taxonomy routes for the Compileo API."""

import logging
import json
import os
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import TaxonomyRepository
from ..core.database import get_db
from ..core.dependencies import get_current_active_user
from ..models.user import User as UserSchema

# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models
class TaxonomyGenerateRequest(BaseModel):
    project_id: str
    name: Optional[str] = None
    documents: List[str]
    depth: int = 3
    generator: str = "gemini"
    domain: str = "general"
    batch_size: int = 10
    category_limits: Optional[List[int]] = None
    specificity_level: int = 1
    processing_mode: str = "fast"  # "fast" (single pass) or "complete" (multi-stage)

class TaxonomyManualCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    project_id: str
    taxonomy: dict

class TaxonomyResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    project_id: str
    categories_count: int
    confidence_score: float
    created_at: datetime
    file_path: str

class TaxonomyListResponse(BaseModel):
    taxonomies: List[TaxonomyResponse]
    total: int

class TaxonomyDetailResponse(BaseModel):
    taxonomy: dict
    metadata: dict
    analytics: dict

class TaxonomyUpdateRequest(BaseModel):
    name: str

class TaxonomyBulkDelete(BaseModel):
    taxonomy_ids: List[str]

class TaxonomyExtendRequest(BaseModel):
    taxonomy_id: Optional[str] = None
    taxonomy_data: Optional[dict] = None
    project_id: Optional[str] = None
    category_path: Optional[str] = None  # Path to specific category to extend (e.g., "cat_0_1_2")
    additional_depth: int = 2
    generator: str = "gemini"
    domain: str = "general"
    batch_size: Optional[int] = None
    category_limits: Optional[List[int]] = None
    specificity_level: int = 1
    documents: Optional[List[str]] = None
    processing_mode: str = "fast"  # "fast" (single pass) or "complete" (multi-stage)

# Helper functions for taxonomy analysis
def calculate_taxonomy_depth(node: Dict[str, Any], current_depth: int = 0) -> int:
    """Calculate the maximum depth of a taxonomy tree."""
    if not node.get('children'):
        return current_depth

    max_child_depth = current_depth
    for child in node['children']:
        child_depth = calculate_taxonomy_depth(child, current_depth + 1)
        max_child_depth = max(max_child_depth, child_depth)

    return max_child_depth

def count_taxonomy_nodes(node: Dict[str, Any]) -> int:
    """Count total number of nodes in taxonomy tree."""
    count = 1  # Count current node
    for child in node.get('children', []):
        count += count_taxonomy_nodes(child)
    return count

def calculate_avg_confidence(node: Dict[str, Any]) -> float:
    """Calculate average confidence score across all nodes."""
    confidences = []

    def collect_confidences(n: Dict[str, Any]):
        if 'confidence_threshold' in n:
            confidences.append(n['confidence_threshold'])
        for child in n.get('children', []):
            collect_confidences(child)

    collect_confidences(node)
    return round(sum(confidences) / len(confidences), 3) if confidences else 0.0

@router.post("/")
async def create_manual_taxonomy(
    request: TaxonomyManualCreateRequest,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Create a new manual taxonomy for current user."""
    try:
        # Validate project exists and belongs to user
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?", (request.project_id, current_user.id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")

        # Validate taxonomy structure
        if not request.taxonomy or not isinstance(request.taxonomy, dict):
            raise HTTPException(status_code=400, detail="Invalid taxonomy structure")

        # Add hierarchical IDs to manually created taxonomy (same as AI-generated)
        from ...features.taxonomy.response_parser import TaxonomyResponseParser
        taxonomy_with_ids = TaxonomyResponseParser._add_hierarchical_ids(request.taxonomy, request.project_id, 1)

        # Calculate analytics
        categories_count = count_taxonomy_nodes(taxonomy_with_ids)
        confidence_score = calculate_avg_confidence(taxonomy_with_ids)

        # Save the taxonomy
        taxonomy_dir = Path(f"storage/{current_user.id}/taxonomy/{request.project_id}")
        taxonomy_dir.mkdir(parents=True, exist_ok=True)
        taxonomy_file_path = taxonomy_dir / f"manual_taxonomy_{uuid.uuid4()}.json"

        taxonomy_data = {
            "taxonomy": taxonomy_with_ids,
            "owner_id": current_user.id,
            "generation_metadata": {
                "type": "manual",
                "confidence_score": confidence_score,
                "created_manually": True
            },
            "analytics": {
                "depth_analysis": {
                    "total_categories": categories_count,
                    "max_depth": calculate_taxonomy_depth(taxonomy_with_ids)
                }
            }
        }

        with open(taxonomy_file_path, 'w') as f:
            json.dump(taxonomy_data, f, indent=2)

        # Save taxonomy metadata to database
        from ...storage.src.project.database_repositories import TaxonomyRepository
        taxonomy_repo = TaxonomyRepository(current_user.id)
        taxonomy_record_id = taxonomy_repo.create_taxonomy({
            'id': str(uuid.uuid4()),
            'owner_id': current_user.id,
            'project_id': request.project_id,
            'name': request.name,
            'structure': taxonomy_data,
            'file_path': str(taxonomy_file_path),
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        })
        db.commit()

        return TaxonomyResponse(
            id=taxonomy_record_id,
            name=request.name,
            description=request.description,
            project_id=request.project_id,
            categories_count=categories_count,
            confidence_score=confidence_score,
            created_at=datetime.now(),
            file_path=str(taxonomy_file_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create manual taxonomy: {str(e)}")

@router.post("/generate")
async def generate_taxonomy(
    request: TaxonomyGenerateRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Generate a new taxonomy for current user."""
    try:
        # Debug logging
        logger.debug(f"DEBUG: Received taxonomy generation request: {request.dict()}")
        # Validate project exists and belongs to user
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?", (request.project_id, current_user.id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")

        # Create job record
        job_id = str(uuid.uuid4())

        # Load chunks from documents - load ALL chunk files for each document belonging to user
        chunks = []
        logger.debug(f"DEBUG: Loading chunks for documents: {request.documents}")
        for doc_id in request.documents:
            logger.debug(f"DEBUG: Processing document ID: {doc_id}")

            # Find ALL chunk records for this document and user
            cursor.execute("""
                SELECT c.id, c.file_path, c.chunk_strategy
                FROM chunks c
                WHERE c.document_id = ? AND c.owner_id = ?
                ORDER BY c.created_at DESC
            """, (doc_id, current_user.id))

            chunk_records = cursor.fetchall()
            logger.debug(f"DEBUG: Found {len(chunk_records)} chunk records for doc {doc_id}")

            for i, chunk_record in enumerate(chunk_records):
                record_id, chunk_file_path, chunk_strategy = chunk_record
                logger.debug(f"DEBUG: Processing chunk {i+1}/{len(chunk_records)}: {chunk_file_path} (strategy: {chunk_strategy})")

                try:
                    if chunk_file_path:
                        # Resolve absolute path using FileManager
                        from ...storage.src.project.file_manager import FileManager
                        storage_base = FileManager.get_storage_base_path()
                        abs_file_path = storage_base / chunk_file_path
                        file_exists = abs_file_path.exists()
                        logger.debug(f"DEBUG: File exists: {abs_file_path} ({file_exists})")
                        if file_exists:
                            # Handle JSON chunk files (contains array of chunks)
                            if chunk_file_path.endswith('.json'):
                                with open(abs_file_path, 'r') as f:
                                    chunk_data = json.load(f)
                                    logger.debug(f"DEBUG: Loaded JSON chunk data type: {type(chunk_data)}")
                                    if isinstance(chunk_data, list):
                                        chunks.extend(chunk_data)
                                        logger.debug(f"DEBUG: Extended chunks with {len(chunk_data)} items from JSON list")
                                    elif isinstance(chunk_data, dict) and 'chunks' in chunk_data:
                                        chunks.extend(chunk_data['chunks'])
                                        logger.debug(f"DEBUG: Extended chunks with {len(chunk_data['chunks'])} items from JSON dict")

                            # Handle individual text chunk files
                            elif chunk_file_path.endswith(('.txt', '.md')):
                                logger.debug(f"DEBUG: Loading individual text chunk file: {abs_file_path}")
                                with open(abs_file_path, 'r', encoding='utf-8') as f:
                                    chunk_text = f.read().strip()
                                    if chunk_text:  # Only add non-empty chunks
                                        chunks.append(chunk_text)
                                        logger.debug(f"DEBUG: Added 1 text chunk with {len(chunk_text)} characters")

                            else:
                                logger.debug(f"DEBUG: Unsupported chunk file extension: {chunk_file_path}")

                        else:
                            logger.debug(f"DEBUG: Chunk file does not exist: {abs_file_path}")
                            # Mark the record as deleted since file is missing
                            cursor.execute("""
                                UPDATE chunks SET status = 'deleted'
                                WHERE id = ?
                            """, (record_id,))
                            db.commit()
                            logger.debug(f"DEBUG: Marked missing chunk record as deleted: {record_id}")

                    else:
                        logger.debug(f"DEBUG: Chunk file path is None for record: {record_id}")
                        # Mark the record as deleted since file path is missing
                        cursor.execute("""
                            UPDATE chunks SET status = 'deleted'
                            WHERE id = ?
                        """, (record_id,))
                        db.commit()
                        logger.debug(f"DEBUG: Marked chunk record with null path as deleted: {record_id}")

                except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
                    logger.debug(f"DEBUG: Error loading chunk file {chunk_file_path}: {e}")
                    # Mark the record as deleted since it's corrupted
                    cursor.execute("""
                        UPDATE chunks SET status = 'deleted'
                        WHERE id = ?
                    """, (record_id,))
                    db.commit()
                    logger.debug(f"DEBUG: Marked corrupted chunk record as deleted: {record_id}")
                    continue

            if not chunk_records:
                logger.debug(f"DEBUG: No chunk records found for document {doc_id}")

        logger.debug(f"DEBUG: Total chunks loaded: {len(chunks)}")
        if not chunks:
            logger.debug("DEBUG: No chunks found - raising 400 error")
            raise HTTPException(
                status_code=400,
                detail="No chunks found for the selected documents. Please chunk the documents first using the document processing endpoint before generating a taxonomy."
            )

        logger.debug(f"DEBUG: About to get API key for generator: {request.generator}")
        # Get API key from settings based on generator type (skip for Ollama)
        from ...core.settings import BackendSettings
        user_settings = BackendSettings(owner_id=current_user.id)
        
        api_key = None
        if request.generator == "grok":
            api_key = user_settings.get_setting("grok_api_key")
        elif request.generator == "gemini":
            api_key = user_settings.get_setting("gemini_api_key")
        elif request.generator == "openai":
            api_key = user_settings.get_setting("openai_api_key")
        # Ollama doesn't require an API key

        if request.generator != "ollama" and not api_key:
            raise HTTPException(status_code=400, detail=f"API key not configured for {request.generator}")

        # Get document name for taxonomy root (use first document name without extension)
        document_name = None
        if request.documents:
            # Use the first document ID to get the name
            cursor.execute("SELECT file_name FROM documents WHERE id = ?", (request.documents[0],))
            doc_record = cursor.fetchone()
            if doc_record:
                file_name = doc_record[0]
                # Remove extension from file name
                document_name = os.path.splitext(file_name)[0] if file_name else None

        logger.debug(f"DEBUG: Using document name for taxonomy root: {document_name}")
        # Generate taxonomy based on generator type
        logger.debug(f"DEBUG: Starting taxonomy generation with {request.generator}, mode: {request.processing_mode}")
        try:
            # Get specific model from settings based on provider
            model_name = None
            if request.generator == "gemini":
                model_name = user_settings.get_taxonomy_gemini_model()
            elif request.generator == "grok":
                model_name = user_settings.get_taxonomy_grok_model()
            elif request.generator == "openai":
                model_name = user_settings.get_taxonomy_openai_model()
            elif request.generator == "ollama":
                model_name = user_settings.get_taxonomy_ollama_model()

            # Prepare arguments for generation
            gen_args = {
                "chunks": chunks,
                "domain": request.domain,
                "depth": request.depth,
                "batch_size": request.batch_size,
                "category_limits": request.category_limits,
                "specificity_level": request.specificity_level,
                "processing_mode": request.processing_mode
            }

            if request.generator == "grok":
                logger.debug("DEBUG: Using Grok generator")
                from ...features.taxonomy.grok_generator import GrokTaxonomyGenerator
                generator = GrokTaxonomyGenerator(str(api_key), model=model_name)
                taxonomy_result = generator.generate_taxonomy(**gen_args)
            elif request.generator == "gemini":
                logger.debug("DEBUG: Using Gemini generator")
                from ...features.taxonomy.gemini_generator import GeminiTaxonomyGenerator
                generator = GeminiTaxonomyGenerator(str(api_key), model=model_name)
                taxonomy_result = generator.generate_taxonomy(**gen_args)
            elif request.generator == "openai":
                logger.debug("DEBUG: Using OpenAI generator")
                from ...features.taxonomy.openai_generator import OpenAITaxonomyGenerator
                generator = OpenAITaxonomyGenerator(str(api_key), model=model_name)
                taxonomy_result = generator.generate_taxonomy(**gen_args)
            elif request.generator == "ollama":
                logger.debug(f"DEBUG: Using Ollama generator - about to initialize")
                from ...features.taxonomy.ollama_generator import OllamaTaxonomyGenerator
                generator = OllamaTaxonomyGenerator(model=model_name)
                logger.debug(f"DEBUG: Ollama generator initialized, calling generate_taxonomy")
                taxonomy_result = generator.generate_taxonomy(**gen_args)
                logger.debug(f"DEBUG: Ollama generate_taxonomy completed, result keys: {list(taxonomy_result.keys()) if taxonomy_result else 'None'}")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported generator: {request.generator}")

            logger.debug(f"DEBUG: Taxonomy generation completed successfully")

            # Modify the taxonomy root name to use document name if available
            if document_name and 'taxonomy' in taxonomy_result:
                taxonomy_result['taxonomy']['name'] = document_name
                logger.debug(f"DEBUG: Set taxonomy root name to: {document_name}")

        except Exception as gen_error:
            logger.debug(f"DEBUG: Taxonomy generation failed: {gen_error}")
            raise HTTPException(status_code=400, detail=f"Taxonomy generation failed: {str(gen_error)}")

        # Save the taxonomy result
        taxonomy_dir = Path(f"storage/{current_user.id}/taxonomy/{request.project_id}")
        taxonomy_dir.mkdir(parents=True, exist_ok=True)
        taxonomy_file_path = taxonomy_dir / f"ai_taxonomy_{job_id}.json"

        # Don't overwrite the taxonomy root name - keep the AI-generated name
        # The user-provided name is stored separately in the database
        taxonomy_result['owner_id'] = current_user.id
        with open(taxonomy_file_path, 'w') as f:
            json.dump(taxonomy_result, f, indent=2)

        # Save taxonomy metadata to database
        from ...storage.src.project.database_repositories import TaxonomyRepository
        taxonomy_repo = TaxonomyRepository(current_user.id)
        taxonomy_record_id = taxonomy_repo.create_taxonomy({
            'id': str(uuid.uuid4()),
            'owner_id': current_user.id,
            'project_id': request.project_id,
            'name': request.name or taxonomy_result.get('taxonomy', {}).get('name', 'AI Generated Taxonomy'),
            'structure': taxonomy_result,
            'file_path': str(taxonomy_file_path),
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        })
        db.commit()

        return {
            "id": taxonomy_record_id,
            "name": request.name if request.name else taxonomy_result.get('taxonomy', {}).get('name', 'AI Generated Taxonomy'),
            "description": taxonomy_result.get('taxonomy', {}).get('description', f'AI-generated taxonomy using {request.generator}'),
            "project_id": request.project_id,
            "categories_count": taxonomy_result.get('analytics', {}).get('depth_analysis', {}).get('total_categories', 0),
            "confidence_score": taxonomy_result.get('generation_metadata', {}).get('confidence_score', 0.0),
            "created_at": datetime.now(),
            "file_path": str(taxonomy_file_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate taxonomy: {str(e)}")

@router.get("", response_model=TaxonomyListResponse)
async def list_taxonomies(
    project_id: Optional[str] = None, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """List all available taxonomies for current user."""
    try:
        taxonomy_repo = TaxonomyRepository(current_user.id)
        if project_id:
            taxonomies_raw = taxonomy_repo.get_taxonomies_by_project(project_id)
        else:
            # For all projects, we need to get all taxonomies for this user
            cursor = db.cursor()
            cursor.execute("SELECT id FROM projects WHERE owner_id = ?", (current_user.id,))
            project_ids = [row[0] for row in cursor.fetchall()]
            taxonomies_raw = []
            for pid in project_ids:
                taxonomies_raw.extend(taxonomy_repo.get_taxonomies_by_project(pid))

        taxonomies = []
        for tax_record in taxonomies_raw:
            try:
                # Parse taxonomy data from database (stored in 'structure' column)
                taxonomy_data_str = tax_record.get('structure')
                if not taxonomy_data_str:
                    continue

                tax_data = json.loads(taxonomy_data_str)
                taxonomy_info = tax_data.get('taxonomy', {})
                metadata = tax_data.get('generation_metadata', {})
                analytics = tax_data.get('analytics', {})

                taxonomies.append(TaxonomyResponse(
                    id=tax_record['id'],
                    name=tax_record.get('name', taxonomy_info.get('name', 'Unknown')),
                    description=taxonomy_info.get('description', ''),
                    project_id=tax_record['project_id'],
                    categories_count=analytics.get('depth_analysis', {}).get('total_categories', 0),
                    confidence_score=metadata.get('confidence_score', 0.0) if metadata else 0.0,
                    created_at=tax_record['created_at'],
                    file_path=''  # No file path in new schema
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        return TaxonomyListResponse(taxonomies=taxonomies, total=len(taxonomies))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list taxonomies: {str(e)}")

@router.get("/{taxonomy_id}", response_model=TaxonomyDetailResponse)
async def get_taxonomy(
    taxonomy_id: str, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Get detailed taxonomy information for current user."""
    try:
        taxonomy_repo = TaxonomyRepository(current_user.id)
        taxonomy_data = taxonomy_repo.get_taxonomy_by_id(taxonomy_id)
        if not taxonomy_data:
            raise HTTPException(status_code=404, detail=f"Taxonomy with ID {taxonomy_id} not found")

        # Parse the taxonomy data from JSON (stored in 'structure' column)
        tax_data = json.loads(taxonomy_data['structure'])
        return TaxonomyDetailResponse(
            taxonomy=tax_data.get('taxonomy', {}),
            metadata=tax_data.get('generation_metadata', {}),
            analytics=tax_data.get('analytics', {})
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid taxonomy data format")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get taxonomy: {str(e)}")

@router.put("/{taxonomy_id}", response_model=TaxonomyResponse)
async def update_taxonomy(
    taxonomy_id: str, 
    update_request: TaxonomyUpdateRequest, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Update taxonomy details for current user."""
    try:
        taxonomy_repo = TaxonomyRepository(current_user.id)

        # Check if taxonomy exists and belongs to user
        taxonomy_data = taxonomy_repo.get_taxonomy_by_id(taxonomy_id)
        if not taxonomy_data:
            raise HTTPException(status_code=404, detail=f"Taxonomy with ID {taxonomy_id} not found")

        # Update the taxonomy name
        cursor = db.cursor()
        cursor.execute("UPDATE taxonomies SET name = ? WHERE id = ? AND owner_id = ?", (update_request.name, taxonomy_id, current_user.id))
        db.commit()

        # Get updated taxonomy info
        return await get_taxonomy_summary(taxonomy_id, db, current_user)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update taxonomy: {str(e)}")

@router.delete("/{taxonomy_id}")
async def delete_taxonomy(
    taxonomy_id: str, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Delete a taxonomy for current user."""
    try:
        logger.debug(f"DEBUG: delete_taxonomy called taxonomy_id={taxonomy_id} owner_id={current_user.id}")
        taxonomy_repo = TaxonomyRepository(current_user.id)

        # Check if taxonomy exists and belongs to user
        taxonomy_data = taxonomy_repo.get_taxonomy_by_id(taxonomy_id)
        if not taxonomy_data:
            logger.debug(f"DEBUG: taxonomy not found taxonomy_id={taxonomy_id} owner_id={current_user.id}")
            raise HTTPException(status_code=404, detail=f"Taxonomy with ID {taxonomy_id} not found")

        # Delete the taxonomy file from filesystem
        taxonomy_file_path_str = taxonomy_data.get('file_path')
        if taxonomy_file_path_str:
            from pathlib import Path
            taxonomy_file_path = Path(taxonomy_file_path_str)
            logger.debug(f"DEBUG: Attempting to delete taxonomy file: {taxonomy_file_path}")

            if taxonomy_file_path.exists():
                try:
                    taxonomy_file_path.unlink()
                    logger.debug(f"DEBUG: Successfully deleted taxonomy file: {taxonomy_file_path}")
                except Exception as file_error:
                    logger.warning(f"Warning: Failed to delete taxonomy file {taxonomy_file_path}: {file_error}")
                    # Continue with database deletion even if file deletion fails
            else:
                logger.debug(f"DEBUG: Taxonomy file not found at expected location: {taxonomy_file_path}")
        else:
            logger.debug(f"DEBUG: No file_path stored for taxonomy {taxonomy_id}")

        # Delete the taxonomy from database
        deleted = taxonomy_repo.delete_taxonomy(taxonomy_id)
        logger.debug(f"DEBUG: taxonomy_repo.delete_taxonomy({taxonomy_id}) -> {deleted}")
        if not deleted:
            raise HTTPException(status_code=500, detail="Failed to delete taxonomy")

        return {"message": f"Taxonomy {taxonomy_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete taxonomy: {str(e)}")

@router.delete("/")
async def bulk_delete_taxonomies(
    bulk_delete: TaxonomyBulkDelete, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Delete multiple taxonomies for current user."""
    if not bulk_delete.taxonomy_ids:
        raise HTTPException(status_code=400, detail="No taxonomy IDs provided")

    deleted_taxonomies = []
    failed_taxonomies = []

    taxonomy_repo = TaxonomyRepository(current_user.id)

    for taxonomy_id in bulk_delete.taxonomy_ids:
        try:
            # Check if taxonomy exists and belongs to user
            taxonomy_data = taxonomy_repo.get_taxonomy_by_id(taxonomy_id)
            if not taxonomy_data:
                failed_taxonomies.append({"id": taxonomy_id, "error": "Taxonomy not found"})
                continue

            # Delete the taxonomy file from filesystem
            taxonomy_file_path_str = taxonomy_data.get('file_path')
            if taxonomy_file_path_str:
                from pathlib import Path
                taxonomy_file_path = Path(taxonomy_file_path_str)

                if taxonomy_file_path.exists():
                    try:
                        taxonomy_file_path.unlink()
                        logger.debug(f"DEBUG: Deleted taxonomy file: {taxonomy_file_path}")
                    except Exception as file_error:
                        logger.warning(f"Warning: Failed to delete taxonomy file {taxonomy_file_path}: {file_error}")
                        # Continue with database deletion even if file deletion fails

            # Delete the taxonomy from database
            if taxonomy_repo.delete_taxonomy(taxonomy_id):
                deleted_taxonomies.append(taxonomy_id)
            else:
                failed_taxonomies.append({"id": taxonomy_id, "error": "Failed to delete taxonomy"})

        except Exception as e:
            failed_taxonomies.append({"id": taxonomy_id, "error": str(e)})

    if failed_taxonomies:
        # Some deletions failed, return partial success
        return {
            "message": f"Deleted {len(deleted_taxonomies)} taxonomies successfully, {len(failed_taxonomies)} failed",
            "deleted": deleted_taxonomies,
            "failed": failed_taxonomies
        }
    else:
        # All deletions successful
        return {
            "message": f"Successfully deleted {len(deleted_taxonomies)} taxonomies",
            "deleted": deleted_taxonomies
        }

async def get_taxonomy_summary(taxonomy_id: str, db, current_user: UserSchema):
    """Helper function to get taxonomy summary for responses."""
    taxonomy_repo = TaxonomyRepository(current_user.id)
    tax_record = taxonomy_repo.get_taxonomy_by_id(taxonomy_id)
    
    if not tax_record:
        raise HTTPException(status_code=404, detail=f"Taxonomy with ID {taxonomy_id} not found")

    try:
        tax_data = json.loads(tax_record['structure'])
        taxonomy_info = tax_data.get('taxonomy', {})
        metadata = tax_data.get('generation_metadata', {})
        analytics = tax_data.get('analytics', {})

        return TaxonomyResponse(
            id=tax_record['id'],
            name=tax_record.get('name', taxonomy_info.get('name', 'Unknown')),
            description=taxonomy_info.get('description', ''),
            project_id=tax_record['project_id'],
            categories_count=analytics.get('depth_analysis', {}).get('total_categories', 0),
            confidence_score=metadata.get('confidence_score', 0.0),
            created_at=tax_record['created_at'],
            file_path=tax_record.get('file_path', '')
        )
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=500, detail="Invalid taxonomy format")

def get_category_by_path(path: str, taxonomy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get a category from the taxonomy by its path."""
    parts = path.split("_")
    if len(parts) < 2:
        return None

    if parts[0] == "cat":
        # Navigate to the category
        cat_idx = int(parts[1])
        if cat_idx >= len(taxonomy.get("children", [])):
            return None

        current = taxonomy["children"][cat_idx]

        # Navigate deeper if there are more path parts
        for part in parts[2:]:
            if part.isdigit():
                idx = int(part)
                if idx < len(current.get("children", [])):
                    current = current["children"][idx]
                else:
                    return None
            else:
                return None

        return current

    return None

@router.post("/extend")
async def extend_taxonomy(
    request: TaxonomyExtendRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """Extend an existing taxonomy with additional depth levels for current user."""
    logger.debug(f"DEBUG: Received extend request: {request.dict()}")
    # Initialize variables to avoid unbound errors
    file_path = None
    temp_taxonomy_file_path = None
    temp_taxonomy_id = None
    existing_taxonomy = None
    project_id = None
    is_temp_taxonomy = False

    try:
        cursor = db.cursor()  # Initialize cursor at the start

        if request.taxonomy_id:
            # Load existing taxonomy from database and verify ownership
            taxonomy_repo = TaxonomyRepository(current_user.id)
            tax_record = taxonomy_repo.get_taxonomy_by_id(request.taxonomy_id)
            if not tax_record:
                raise HTTPException(status_code=404, detail=f"Taxonomy with ID {request.taxonomy_id} not found")

            file_path = tax_record.get('file_path')
            project_id = tax_record.get('project_id')
            
            # Load existing taxonomy
            existing_tax_data = json.loads(tax_record['structure'])
            existing_taxonomy = existing_tax_data.get('taxonomy', {})
        elif request.taxonomy_data and request.project_id:
            # Use provided taxonomy data directly - verify project ownership
            cursor.execute("SELECT id FROM projects WHERE id = ? AND owner_id = ?", (request.project_id, current_user.id))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")
                
            existing_taxonomy = request.taxonomy_data
            project_id = request.project_id
            is_temp_taxonomy = True

            # Create temporary taxonomy for processing
            temp_taxonomy_data = {
                "taxonomy": existing_taxonomy,
                "owner_id": current_user.id,
                "generation_metadata": {
                    "type": "temporary",
                    "created_for_extension": True
                },
                "analytics": {
                    "depth_analysis": {
                        "total_categories": count_taxonomy_nodes(existing_taxonomy),
                        "max_depth": calculate_taxonomy_depth(existing_taxonomy)
                    }
                }
            }

            # Save temporary taxonomy metadata to database
            from ...storage.src.project.database_repositories import TaxonomyRepository
            taxonomy_repo = TaxonomyRepository(current_user.id)
            temp_taxonomy_id = taxonomy_repo.create_taxonomy({
                'id': str(uuid.uuid4()),
                'owner_id': current_user.id,
                'project_id': project_id,
                'name': "Temporary Taxonomy for Extension",
                'structure': temp_taxonomy_data,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            })
            db.commit()
        else:
            raise HTTPException(status_code=400, detail="Either taxonomy_id or both taxonomy_data and project_id must be provided")

        # Load chunks from project documents
        chunks = []
        
        # Build query based on whether specific documents are selected
        if request.documents:
            # Use placeholders for the document IDs
            placeholders = ','.join(['?'] * len(request.documents))
            query = f"""
                SELECT c.file_path
                FROM chunks c
                WHERE c.document_id IN ({placeholders})
                AND c.owner_id = ?
                AND c.document_id IN (
                    SELECT d.id FROM documents d WHERE d.project_id = ? AND d.owner_id = ?
                )
                ORDER BY c.created_at DESC
            """
            params = request.documents + [current_user.id, project_id, current_user.id]
            logger.debug(f"DEBUG: Loading chunks from specific documents: {request.documents}")
        else:
            query = """
                SELECT c.file_path
                FROM chunks c
                WHERE c.owner_id = ?
                AND c.document_id IN (
                    SELECT d.id FROM documents d WHERE d.project_id = ? AND d.owner_id = ?
                )
                ORDER BY c.created_at DESC
            """
            params = [current_user.id, project_id, current_user.id]
            logger.debug(f"DEBUG: Loading chunks from all project documents")

        cursor.execute(query, params)

        chunk_records = cursor.fetchall()
        logger.debug(f"DEBUG: Found {len(chunk_records)} chunk records")

        for i, chunk_record in enumerate(chunk_records):
            chunk_file_path = chunk_record[0]
            logger.debug(f"DEBUG: Processing chunk {i+1}/{len(chunk_records)}: {chunk_file_path}")

            try:
                if chunk_file_path:
                    # Resolve absolute path using FileManager
                    from ...storage.src.project.file_manager import FileManager
                    storage_base = FileManager.get_storage_base_path()
                    abs_file_path = storage_base / chunk_file_path
                    file_exists = abs_file_path.exists()
                    logger.debug(f"DEBUG: File exists: {abs_file_path} ({file_exists})")

                    if file_exists:
                        # Handle JSON chunk files (contains array of chunks)
                        if chunk_file_path.endswith('.json'):
                            with open(abs_file_path, 'r') as f:
                                chunk_data = json.load(f)
                                logger.debug(f"DEBUG: Loaded JSON chunk data type: {type(chunk_data)}")
                                if isinstance(chunk_data, list):
                                    chunks.extend(chunk_data)
                                    logger.debug(f"DEBUG: Extended chunks with {len(chunk_data)} items from JSON list")
                                elif isinstance(chunk_data, dict) and 'chunks' in chunk_data:
                                    chunks.extend(chunk_data['chunks'])
                                    logger.debug(f"DEBUG: Extended chunks with {len(chunk_data['chunks'])} items from JSON dict")

                        # Handle individual text chunk files
                        elif chunk_file_path.endswith(('.txt', '.md')):
                                logger.debug(f"DEBUG: Loading individual text chunk file: {abs_file_path}")
                                with open(abs_file_path, 'r', encoding='utf-8') as f:
                                    chunk_text = f.read().strip()
                                    if chunk_text:  # Only add non-empty chunks
                                        chunks.append(chunk_text)
                                        logger.debug(f"DEBUG: Added 1 text chunk with {len(chunk_text)} characters")

                        else:
                            logger.debug(f"DEBUG: Unsupported chunk file extension: {chunk_file_path}")

                    else:
                        logger.debug(f"DEBUG: Chunk file does not exist: {abs_file_path}")

                else:
                    logger.debug(f"DEBUG: Chunk file path is None for record {i+1}")

            except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
                logger.debug(f"DEBUG: Error loading chunk file {chunk_file_path}: {e}")
                continue

        logger.debug(f"DEBUG: Total chunks loaded: {len(chunks)}")

        if not chunks:
            # For category extension, use the category itself as context
            if request.taxonomy_data:
                # Use category name and description as context for extension
                chunks = []
                def extract_text_from_category(cat):
                    if 'name' in cat and cat['name']:
                        chunks.append(f"Category: {cat['name']}")
                    if 'description' in cat and cat['description']:
                        chunks.append(f"Description: {cat['description']}")
                    for child in cat.get('children', []):
                        extract_text_from_category(child)
                extract_text_from_category(request.taxonomy_data)
                logger.debug(f"DEBUG: Using category structure as context, found {len(chunks)} text items")
            else:
                raise HTTPException(status_code=400, detail="No chunks found for the project. Please process some documents first to enable taxonomy extension.")

        # Get API key based on generator type
        from ...core.settings import BackendSettings
        user_settings = BackendSettings(owner_id=current_user.id)
        
        api_key = None
        if request.generator == "grok":
            api_key = user_settings.get_setting("grok_api_key")
        elif request.generator == "gemini":
            api_key = user_settings.get_setting("gemini_api_key")
        elif request.generator == "openai":
            api_key = user_settings.get_setting("openai_api_key")

        if request.generator != "ollama" and not api_key:
            raise HTTPException(status_code=400, detail=f"API key not configured for {request.generator}")

        # Get specific model from settings for extension
        model_name = None
        if request.generator == "gemini":
            model_name = user_settings.get_taxonomy_gemini_model()
        elif request.generator == "grok":
            model_name = user_settings.get_taxonomy_grok_model()
        elif request.generator == "openai":
            model_name = user_settings.get_taxonomy_openai_model()
        elif request.generator == "ollama":
            model_name = user_settings.get_taxonomy_ollama_model()

        # Extend taxonomy based on generator type
        try:
            if request.generator == "grok":
                from ...features.taxonomy.grok_generator import GrokTaxonomyGenerator
                generator = GrokTaxonomyGenerator(str(api_key), model=model_name)
                extended_result = generator.extend_taxonomy(
                    existing_taxonomy=existing_taxonomy,
                    chunks=chunks,
                    additional_depth=request.additional_depth,
                    domain=request.domain,
                    batch_size=request.batch_size,
                    category_limits=request.category_limits,
                    specificity_level=request.specificity_level
                )
            elif request.generator == "gemini":
                from ...features.taxonomy.gemini_generator import GeminiTaxonomyGenerator
                generator = GeminiTaxonomyGenerator(str(api_key), model=model_name)
                extended_result = generator.extend_taxonomy(
                    existing_taxonomy=existing_taxonomy,
                    chunks=chunks,
                    additional_depth=request.additional_depth,
                    domain=request.domain,
                    batch_size=request.batch_size,
                    category_limits=request.category_limits,
                    specificity_level=request.specificity_level,
                    processing_mode=request.processing_mode
                )
            elif request.generator == "openai":
                from ...features.taxonomy.openai_generator import OpenAITaxonomyGenerator
                generator = OpenAITaxonomyGenerator(str(api_key), model=model_name)
                extended_result = generator.extend_taxonomy(
                    existing_taxonomy=existing_taxonomy,
                    chunks=chunks,
                    additional_depth=request.additional_depth,
                    domain=request.domain,
                    batch_size=request.batch_size,
                    category_limits=request.category_limits,
                    specificity_level=request.specificity_level
                )
            elif request.generator == "ollama":
                from ...features.taxonomy.ollama_generator import OllamaTaxonomyGenerator
                generator = OllamaTaxonomyGenerator(model=model_name)
                extended_result = generator.extend_taxonomy(
                    existing_taxonomy=existing_taxonomy,
                    chunks=chunks,
                    additional_depth=request.additional_depth,
                    domain=request.domain,
                    sample_size=request.batch_size,
                    category_limits=request.category_limits,
                    specificity_level=request.specificity_level
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported generator: {request.generator}")
        except Exception as gen_error:
            raise HTTPException(status_code=400, detail=f"Taxonomy extension failed: {str(gen_error)}")

        # Handle response based on whether this was a temporary taxonomy or existing one
        if is_temp_taxonomy:
            # Clean up temporary taxonomy
            try:
                if temp_taxonomy_id:
                    taxonomy_repo = TaxonomyRepository(current_user.id)
                    taxonomy_repo.delete_taxonomy(temp_taxonomy_id)
            except Exception as cleanup_error:
                logger.warning(f"Warning: Failed to cleanup temporary taxonomy: {cleanup_error}")

            return {
                "extended_taxonomy": extended_result,
                "processing_mode": request.processing_mode,
                "extension_type": "category_extension"
            }
        else:
            # For existing taxonomies, update in place
            taxonomy_repo = TaxonomyRepository(current_user.id)
            cursor.execute("UPDATE taxonomies SET structure = ?, updated_at = ? WHERE id = ? AND owner_id = ?", 
                          (json.dumps(extended_result), datetime.now(), request.taxonomy_id, current_user.id))
            db.commit()

            return {
                "id": request.taxonomy_id,
                "name": existing_taxonomy.get('name', 'Taxonomy') if existing_taxonomy else 'Taxonomy',
                "description": f"AI-extended taxonomy with {request.additional_depth} additional levels using {request.generator}",
                "project_id": project_id,
                "categories_count": count_taxonomy_nodes(extended_result),
                "confidence_score": calculate_avg_confidence(extended_result),
                "created_at": datetime.now(),
                "file_path": file_path or '',
                "processing_mode": request.processing_mode,
                "extension_type": "taxonomy_extension"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extend taxonomy: {str(e)}")
