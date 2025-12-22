"""Taxonomy routes for the Compileo API."""

import json
import os
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from ...core.logging import get_logger
from ...storage.src.database import get_db_connection
from ...storage.src.project.database_repositories import TaxonomyRepository

# Create router
router = APIRouter()
logger = get_logger(__name__)

# Pydantic models
class TaxonomyGenerateRequest(BaseModel):
    project_id: str
    name: Optional[str] = None
    documents: List[str]
    depth: Optional[int] = 3
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
    name: Optional[str] = None
    taxonomy: Optional[dict] = None
    description: Optional[str] = None

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

# Dependency to get database connection
def get_db():
    return get_db_connection()

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
    db=Depends(get_db)
):
    """Create a new manual taxonomy."""
    try:
        # Validate project exists
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ?", (request.project_id,))
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
        taxonomy_dir = Path(f"storage/taxonomy/{request.project_id}")
        taxonomy_dir.mkdir(parents=True, exist_ok=True)
        taxonomy_file_path = taxonomy_dir / f"manual_taxonomy_{uuid.uuid4()}.json"

        taxonomy_data = {
            "taxonomy": taxonomy_with_ids,
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
        taxonomy_repo = TaxonomyRepository(db)
        taxonomy_record_id = taxonomy_repo.create_taxonomy({
            'id': str(uuid.uuid4()),
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
    db=Depends(get_db)
):
    """Generate a new taxonomy."""
    try:
        # Debug logging
        logger.debug(f"Received taxonomy generation request: {request.dict()}")
        # Validate project exists
        cursor = db.cursor()
        cursor.execute("SELECT name FROM projects WHERE id = ?", (request.project_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Project with ID {request.project_id} not found")

        # Create job record
        job_id = str(uuid.uuid4())

        # Load chunks from documents - load ALL chunk files for each document
        chunks = []
        logger.debug(f"Loading chunks for documents: {request.documents}")
        for doc_id in request.documents:
            logger.debug(f"Processing document ID: {doc_id}")

            # Find ALL chunk records for this document (not just the most recent)
            cursor.execute("""
                SELECT c.id, c.file_path, c.chunk_strategy
                FROM chunks c
                WHERE c.document_id = ?
                ORDER BY c.created_at DESC
            """, (doc_id,))

            chunk_records = cursor.fetchall()
            logger.debug(f"Found {len(chunk_records)} chunk records for doc {doc_id}")

            for i, chunk_record in enumerate(chunk_records):
                record_id, chunk_file_path, chunk_strategy = chunk_record
                logger.debug(f"Processing chunk {i+1}/{len(chunk_records)}: {chunk_file_path} (strategy: {chunk_strategy})")

                try:
                    if chunk_file_path:
                        # Resolve absolute path using FileManager
                        from ...storage.src.project.file_manager import FileManager
                        storage_base = FileManager.get_storage_base_path()
                        abs_file_path = storage_base / chunk_file_path
                        file_exists = abs_file_path.exists()
                        logger.debug(f"File exists: {abs_file_path} ({file_exists})")
                        if file_exists:
                            # Handle JSON chunk files (contains array of chunks)
                            if chunk_file_path.endswith('.json'):
                                with open(abs_file_path, 'r') as f:
                                    chunk_data = json.load(f)
                                    logger.debug(f"Loaded JSON chunk data type: {type(chunk_data)}")
                                    if isinstance(chunk_data, list):
                                        chunks.extend(chunk_data)
                                        logger.debug(f"Extended chunks with {len(chunk_data)} items from JSON list")
                                    elif isinstance(chunk_data, dict) and 'chunks' in chunk_data:
                                        chunks.extend(chunk_data['chunks'])
                                        logger.debug(f"Extended chunks with {len(chunk_data['chunks'])} items from JSON dict")

                            # Handle individual text chunk files
                            elif chunk_file_path.endswith(('.txt', '.md')):
                                logger.debug(f"Loading individual text chunk file: {abs_file_path}")
                                with open(abs_file_path, 'r', encoding='utf-8') as f:
                                    chunk_text = f.read().strip()
                                    if chunk_text:  # Only add non-empty chunks
                                        chunks.append(chunk_text)
                                        logger.debug(f"Added 1 text chunk with {len(chunk_text)} characters")

                            else:
                                logger.debug(f"Unsupported chunk file extension: {chunk_file_path}")

                        else:
                            logger.debug(f"Chunk file does not exist: {abs_file_path}")
                            # Mark the record as deleted since file is missing
                            cursor.execute("""
                                UPDATE chunks SET status = 'deleted'
                                WHERE id = ?
                            """, (record_id,))
                            db.commit()
                            logger.debug(f"Marked missing chunk record as deleted: {record_id}")

                    else:
                        logger.debug(f"Chunk file path is None for record: {record_id}")
                        # Mark the record as deleted since file path is missing
                        cursor.execute("""
                            UPDATE chunks SET status = 'deleted'
                            WHERE id = ?
                        """, (record_id,))
                        db.commit()
                        logger.debug(f"Marked chunk record with null path as deleted: {record_id}")

                except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
                    logger.debug(f"Error loading chunk file {chunk_file_path}: {e}")
                    # Mark the record as deleted since it's corrupted
                    cursor.execute("""
                        UPDATE chunks SET status = 'deleted'
                        WHERE id = ?
                    """, (record_id,))
                    db.commit()
                    logger.debug(f"Marked corrupted chunk record as deleted: {record_id}")
                    continue

            if not chunk_records:
                logger.debug(f"No chunk records found for document {doc_id}")

        logger.debug(f"Total chunks loaded: {len(chunks)}")
        if not chunks:
            logger.debug("No chunks found - raising 400 error")
            raise HTTPException(
                status_code=400,
                detail="No chunks found for the selected documents. Please chunk the documents first using the document processing endpoint before generating a taxonomy."
            )

        logger.debug(f"About to get API key for generator: {request.generator}")
        # Get API key from settings based on generator type (skip for Ollama)
        from ...core.settings import backend_settings
        api_key = None
        if request.generator == "grok":
            api_key = backend_settings.get_setting("grok_api_key")
        elif request.generator == "gemini":
            api_key = backend_settings.get_setting("gemini_api_key")
        elif request.generator == "openai":
            api_key = backend_settings.get_setting("openai_api_key")
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

        logger.debug(f"Using document name for taxonomy root: {document_name}")
        # Generate taxonomy based on generator type
        logger.debug(f"Starting taxonomy generation with {request.generator}, mode: {request.processing_mode}")
        try:
            # Get specific model from settings based on provider
            model_name = None
            if request.generator == "gemini":
                model_name = backend_settings.get_taxonomy_gemini_model()
            elif request.generator == "grok":
                model_name = backend_settings.get_taxonomy_grok_model()
            elif request.generator == "openai":
                model_name = backend_settings.get_taxonomy_openai_model()
            elif request.generator == "ollama":
                model_name = backend_settings.get_taxonomy_ollama_model()

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
                logger.debug("Using Grok generator")
                from ...features.taxonomy.grok_generator import GrokTaxonomyGenerator
                generator = GrokTaxonomyGenerator(str(api_key), model=model_name)
                taxonomy_result = generator.generate_taxonomy(**gen_args)
            elif request.generator == "gemini":
                logger.debug("Using Gemini generator")
                from ...features.taxonomy.gemini_generator import GeminiTaxonomyGenerator
                generator = GeminiTaxonomyGenerator(str(api_key), model=model_name or "gemini-2.0-flash")
                taxonomy_result = generator.generate_taxonomy(**gen_args)
            elif request.generator == "openai":
                logger.debug("Using OpenAI generator")
                from ...features.taxonomy.openai_generator import OpenAITaxonomyGenerator
                generator = OpenAITaxonomyGenerator(str(api_key), model=model_name)
                taxonomy_result = generator.generate_taxonomy(**gen_args)
            elif request.generator == "ollama":
                logger.debug(f"Using Ollama generator - about to initialize")
                from ...features.taxonomy.ollama_generator import OllamaTaxonomyGenerator
                generator = OllamaTaxonomyGenerator(model=model_name)
                logger.debug(f"Ollama generator initialized, calling generate_taxonomy")
                taxonomy_result = generator.generate_taxonomy(**gen_args)
                logger.debug(f"Ollama generate_taxonomy completed, result keys: {list(taxonomy_result.keys()) if taxonomy_result else 'None'}")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported generator: {request.generator}")

            logger.debug(f"Taxonomy generation completed successfully")

            # Store document IDs and generation parameters in metadata for future regeneration
            if 'generation_metadata' not in taxonomy_result:
                taxonomy_result['generation_metadata'] = {}
            taxonomy_result['generation_metadata']['source_document_ids'] = request.documents
            taxonomy_result['generation_metadata']['original_parameters'] = request.dict()

            # Modify the taxonomy root name to use document name if available
            if document_name and 'taxonomy' in taxonomy_result:
                taxonomy_result['taxonomy']['name'] = document_name
                logger.debug(f"Set taxonomy root name to: {document_name}")

        except Exception as gen_error:
            logger.error(f"Taxonomy generation failed: {gen_error}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Taxonomy generation failed: {str(gen_error)}")

        # Save the taxonomy result
        taxonomy_dir = Path(f"storage/taxonomy/{request.project_id}")
        taxonomy_dir.mkdir(parents=True, exist_ok=True)
        taxonomy_file_path = taxonomy_dir / f"ai_taxonomy_{job_id}.json"

        # Don't overwrite the taxonomy root name - keep the AI-generated name
        # The user-provided name is stored separately in the database
        with open(taxonomy_file_path, 'w') as f:
            json.dump(taxonomy_result, f, indent=2)

        # Save taxonomy metadata to database
        from ...storage.src.project.database_repositories import TaxonomyRepository
        taxonomy_repo = TaxonomyRepository(db)
        taxonomy_record_id = taxonomy_repo.create_taxonomy({
            'id': str(uuid.uuid4()),
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
async def list_taxonomies(project_id: Optional[str] = None, db=Depends(get_db)):
    """List all available taxonomies."""
    try:
        taxonomy_repo = TaxonomyRepository(db)
        if project_id:
            taxonomies_raw = taxonomy_repo.get_taxonomies_by_project(project_id)
        else:
            # For all projects, we need to get all taxonomies
            cursor = db.cursor()
            cursor.execute("SELECT id FROM projects")
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
async def get_taxonomy(taxonomy_id: str, db=Depends(get_db)):
    """Get detailed taxonomy information."""
    try:
        taxonomy_repo = TaxonomyRepository(db)
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
async def update_taxonomy(taxonomy_id: str, update_request: TaxonomyUpdateRequest, db=Depends(get_db)):
    """Update taxonomy details."""
    try:
        taxonomy_repo = TaxonomyRepository(db)

        # Check if taxonomy exists
        existing_record = taxonomy_repo.get_taxonomy_by_id(taxonomy_id)
        if not existing_record:
            raise HTTPException(status_code=404, detail=f"Taxonomy with ID {taxonomy_id} not found")

        # Prepare update data
        update_data = {}
        if update_request.name:
            update_data['name'] = update_request.name
            
        if update_request.taxonomy:
            # If structure is updated, we also need to update the file if it exists
            # and update categories_count etc in the response
            # But the repository handles the DB JSON structure
            
            # Wrap taxonomy for storage compatibility if it's just the children
            if "children" in update_request.taxonomy:
                 new_structure = update_request.taxonomy
            else:
                 # It might be the full taxonomy result or just categories
                 # We try to keep it consistent with create_manual_taxonomy
                 from ...features.taxonomy.response_parser import TaxonomyResponseParser
                 # We don't want to re-run AI but we might want to ensure IDs
                 new_structure = TaxonomyResponseParser._add_hierarchical_ids(update_request.taxonomy, existing_record['project_id'], 1)

            # Re-calculate analytics if structure changed
            categories_count = count_taxonomy_nodes(new_structure)
            confidence_score = calculate_avg_confidence(new_structure)
            
            full_taxonomy_data = {
                "taxonomy": new_structure,
                "generation_metadata": {
                    "type": "manual_update",
                    "confidence_score": confidence_score,
                    "updated_manually": True
                },
                "analytics": {
                    "depth_analysis": {
                        "total_categories": categories_count,
                        "max_depth": calculate_taxonomy_depth(new_structure)
                    }
                }
            }
            update_data['structure'] = full_taxonomy_data
            
            # Update the physical file if it exists
            if existing_record.get('file_path') and os.path.exists(existing_record['file_path']):
                with open(existing_record['file_path'], 'w') as f:
                    json.dump(full_taxonomy_data, f, indent=2)

        # Update the database
        if not taxonomy_repo.update_taxonomy(taxonomy_id, update_data):
            raise HTTPException(status_code=500, detail="Failed to update taxonomy in database")

        # Get updated taxonomy info
        # We need to return a TaxonomyResponse which requires analytics from the JSON
        updated_record = taxonomy_repo.get_taxonomy_by_id(taxonomy_id)
        if not updated_record:
            raise HTTPException(status_code=500, detail="Updated record not found")
            
        tax_data = json.loads(updated_record['structure'])
        analytics = tax_data.get('analytics', {})
        metadata = tax_data.get('generation_metadata', {})
        
        return TaxonomyResponse(
            id=updated_record['id'],
            name=updated_record['name'],
            description=update_request.description or tax_data.get('taxonomy', {}).get('description', ''),
            project_id=updated_record['project_id'],
            categories_count=analytics.get('depth_analysis', {}).get('total_categories', 0),
            confidence_score=metadata.get('confidence_score', 0.0),
            created_at=updated_record['created_at'],
            file_path=updated_record.get('file_path', '')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update taxonomy failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update taxonomy: {str(e)}")

@router.delete("/{taxonomy_id}")
async def delete_taxonomy(taxonomy_id: str, db=Depends(get_db)):
    """Delete a taxonomy."""
    try:
        taxonomy_repo = TaxonomyRepository(db)

        # Check if taxonomy exists
        taxonomy_data = taxonomy_repo.get_taxonomy_by_id(taxonomy_id)
        if not taxonomy_data:
            raise HTTPException(status_code=404, detail=f"Taxonomy with ID {taxonomy_id} not found")

        # Delete the taxonomy file from filesystem
        taxonomy_file_path_str = taxonomy_data.get('file_path')
        if taxonomy_file_path_str:
            from pathlib import Path
            taxonomy_file_path = Path(taxonomy_file_path_str)
            logger.debug(f"Attempting to delete taxonomy file: {taxonomy_file_path}")

            if taxonomy_file_path.exists():
                try:
                    taxonomy_file_path.unlink()
                    logger.debug(f"Successfully deleted taxonomy file: {taxonomy_file_path}")
                except Exception as file_error:
                    logger.warning(f"Failed to delete taxonomy file {taxonomy_file_path}: {file_error}")
                    # Continue with database deletion even if file deletion fails
            else:
                logger.debug(f"Taxonomy file not found at expected location: {taxonomy_file_path}")
        else:
            logger.debug(f"No file_path stored for taxonomy {taxonomy_id}")

        # Delete the taxonomy from database
        if not taxonomy_repo.delete_taxonomy(taxonomy_id):
            raise HTTPException(status_code=500, detail="Failed to delete taxonomy")

        return {"message": f"Taxonomy {taxonomy_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete taxonomy: {str(e)}")

@router.delete("/")
async def bulk_delete_taxonomies(bulk_delete: TaxonomyBulkDelete, db=Depends(get_db)):
    """Delete multiple taxonomies."""
    if not bulk_delete.taxonomy_ids:
        raise HTTPException(status_code=400, detail="No taxonomy IDs provided")

    deleted_taxonomies = []
    failed_taxonomies = []

    taxonomy_repo = TaxonomyRepository(db)

    for taxonomy_id in bulk_delete.taxonomy_ids:
        try:
            # Check if taxonomy exists
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
                        logger.debug(f"Deleted taxonomy file: {taxonomy_file_path}")
                    except Exception as file_error:
                        logger.warning(f"Failed to delete taxonomy file {taxonomy_file_path}: {file_error}")
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

async def get_taxonomy_summary(taxonomy_id: str, db):
    """Helper function to get taxonomy summary for responses."""
    cursor = db.cursor()
    cursor.execute("""
        SELECT po.id, po.project_id, po.output_file_path, po.created_at
        FROM processed_outputs po
        WHERE po.id = ?
    """, (taxonomy_id,))

    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Taxonomy with ID {taxonomy_id} not found")

    tax_id, project_id, file_path, created_at = row

    try:
        with open(file_path, 'r') as f:
            tax_data = json.load(f)
        taxonomy_info = tax_data.get('taxonomy', {})
        metadata = tax_data.get('generation_metadata', {})
        analytics = tax_data.get('analytics', {})

        return TaxonomyResponse(
            id=tax_id,
            name=taxonomy_info.get('name', 'Unknown'),
            description=taxonomy_info.get('description', ''),
            project_id=project_id,
            categories_count=analytics.get('depth_analysis', {}).get('total_categories', 0),
            confidence_score=metadata.get('confidence_score', 0.0),
            created_at=created_at,
            file_path=file_path
        )
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        raise HTTPException(status_code=500, detail="Invalid taxonomy file format")

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
    db=Depends(get_db)
):
    """Extend an existing taxonomy with additional depth levels."""
    logger.debug(f"Received extend request: {request.dict()}")
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
            # Load existing taxonomy from database
            cursor.execute("SELECT output_file_path, project_id FROM processed_outputs WHERE id = ?", (request.taxonomy_id,))
            tax_record = cursor.fetchone()
            if not tax_record:
                raise HTTPException(status_code=404, detail=f"Taxonomy with ID {request.taxonomy_id} not found")

            file_path, project_id = tax_record
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"Taxonomy file not found: {file_path}")

            # Load existing taxonomy
            with open(file_path, 'r') as f:
                existing_tax_data = json.load(f)

            existing_taxonomy = existing_tax_data.get('taxonomy', {})
        elif request.taxonomy_data and request.project_id:
            # Use provided taxonomy data directly - create temporary taxonomy for processing
            existing_taxonomy = request.taxonomy_data
            project_id = request.project_id
            is_temp_taxonomy = True

            # Create temporary taxonomy for processing
            temp_taxonomy_data = {
                "taxonomy": existing_taxonomy,
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

            # Save temporary taxonomy
            temp_taxonomy_dir = Path(f"storage/taxonomy/{project_id}")
            temp_taxonomy_dir.mkdir(parents=True, exist_ok=True)
            temp_taxonomy_file_path = temp_taxonomy_dir / f"temp_taxonomy_{uuid.uuid4()}.json"

            with open(temp_taxonomy_file_path, 'w') as f:
                json.dump(temp_taxonomy_data, f, indent=2)

            # Save temporary taxonomy metadata to database
            from ...storage.src.project.database_repositories import TaxonomyRepository
            taxonomy_repo = TaxonomyRepository(db)
            temp_taxonomy_id = taxonomy_repo.create_taxonomy({
                'id': str(uuid.uuid4()),
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
                AND c.document_id IN (
                    SELECT d.id FROM documents d WHERE d.project_id = ?
                )
                ORDER BY c.created_at DESC
            """
            params = request.documents + [project_id]
            logger.debug(f"Loading chunks from specific documents: {request.documents}")
        else:
            query = """
                SELECT c.file_path
                FROM chunks c
                WHERE c.document_id IN (
                    SELECT d.id FROM documents d WHERE d.project_id = ?
                )
                ORDER BY c.created_at DESC
            """
            params = [project_id]
            logger.debug(f"Loading chunks from all project documents")

        cursor.execute(query, params)

        chunk_records = cursor.fetchall()
        logger.debug(f"Found {len(chunk_records)} chunk records")

        for i, chunk_record in enumerate(chunk_records):
            chunk_file_path = chunk_record[0]
            logger.debug(f"Processing chunk {i+1}/{len(chunk_records)}: {chunk_file_path}")

            try:
                if chunk_file_path:
                    # Resolve absolute path using FileManager
                    from ...storage.src.project.file_manager import FileManager
                    storage_base = FileManager.get_storage_base_path()
                    abs_file_path = storage_base / chunk_file_path
                    file_exists = abs_file_path.exists()
                    logger.debug(f"File exists: {abs_file_path} ({file_exists})")

                    if file_exists:
                        # Handle JSON chunk files (contains array of chunks)
                        if chunk_file_path.endswith('.json'):
                            with open(abs_file_path, 'r') as f:
                                chunk_data = json.load(f)
                                logger.debug(f"Loaded JSON chunk data type: {type(chunk_data)}")
                                if isinstance(chunk_data, list):
                                    chunks.extend(chunk_data)
                                    logger.debug(f"Extended chunks with {len(chunk_data)} items from JSON list")
                                elif isinstance(chunk_data, dict) and 'chunks' in chunk_data:
                                    chunks.extend(chunk_data['chunks'])
                                    logger.debug(f"Extended chunks with {len(chunk_data['chunks'])} items from JSON dict")

                        # Handle individual text chunk files
                        elif chunk_file_path.endswith(('.txt', '.md')):
                            logger.debug(f"Loading individual text chunk file: {abs_file_path}")
                            with open(abs_file_path, 'r', encoding='utf-8') as f:
                                chunk_text = f.read().strip()
                                if chunk_text:  # Only add non-empty chunks
                                    chunks.append(chunk_text)
                                    logger.debug(f"Added 1 text chunk with {len(chunk_text)} characters")

                        else:
                            logger.debug(f"Unsupported chunk file extension: {chunk_file_path}")

                    else:
                        logger.debug(f"Chunk file does not exist: {abs_file_path}")

                else:
                    logger.debug(f"Chunk file path is None for record {i+1}")

            except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
                logger.debug(f"Error loading chunk file {chunk_file_path}: {e}")
                continue

        logger.debug(f"Total chunks loaded: {len(chunks)}")

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
                logger.debug(f"Using category structure as context, found {len(chunks)} text items")
            else:
                raise HTTPException(status_code=400, detail="No chunks found for the project. Please process some documents first to enable taxonomy extension.")

        # Get API key based on generator type (skip for Ollama)
        from ...core.settings import backend_settings
        api_key = None
        if request.generator == "grok":
            api_key = backend_settings.get_setting("grok_api_key")
        elif request.generator == "gemini":
            api_key = backend_settings.get_setting("gemini_api_key")
        elif request.generator == "openai":
            api_key = backend_settings.get_setting("openai_api_key")
        # Ollama doesn't require an API key

        if request.generator != "ollama" and not api_key:
            raise HTTPException(status_code=400, detail=f"API key not configured for {request.generator}")

        # Ensure api_key is a string for non-ollama generators
        if request.generator != "ollama" and not isinstance(api_key, str):
            raise HTTPException(status_code=400, detail=f"Invalid API key configuration for {request.generator}")

        # Get specific model from settings for extension
        model_name = None
        if request.generator == "gemini":
            model_name = backend_settings.get_taxonomy_gemini_model()
        elif request.generator == "grok":
            model_name = backend_settings.get_taxonomy_grok_model()
        elif request.generator == "openai":
            model_name = backend_settings.get_taxonomy_openai_model()
        elif request.generator == "ollama":
            model_name = backend_settings.get_taxonomy_ollama_model()

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
                generator = GeminiTaxonomyGenerator(str(api_key), model=model_name or "gemini-2.0-flash")
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
            # For temporary taxonomies (category extensions), return the extended taxonomy data directly
            # Clean up temporary taxonomy
            try:
                if temp_taxonomy_file_path and os.path.exists(temp_taxonomy_file_path):
                    os.remove(temp_taxonomy_file_path)
                if temp_taxonomy_id:
                    from ...storage.src.project.database_repositories import TaxonomyRepository
                    taxonomy_repo = TaxonomyRepository(db)
                    taxonomy_repo.delete_taxonomy(temp_taxonomy_id)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary taxonomy: {cleanup_error}")

            return {
                "extended_taxonomy": extended_result,
                "processing_mode": request.processing_mode,
                "extension_type": "category_extension"
            }
        else:
            # For existing taxonomies, update in place
            # Overwrite the existing taxonomy file with the extended version
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(extended_result, f, indent=2)

                # Update the database record's created_at to reflect modification time
                cursor.execute("""
                    UPDATE processed_outputs
                    SET created_at = ?
                    WHERE id = ?
                """, (datetime.now(), request.taxonomy_id))
                db.commit()

                return {
                    "id": request.taxonomy_id,
                    "name": existing_taxonomy.get('name', 'Taxonomy') if existing_taxonomy else 'Taxonomy',
                    "description": f"AI-extended taxonomy with {request.additional_depth} additional levels using {request.generator}",
                    "project_id": project_id,
                    "categories_count": extended_result.get('analytics', {}).get('depth_analysis', {}).get('total_categories', 0),
                    "confidence_score": extended_result.get('generation_metadata', {}).get('confidence_score', 0.0),
                    "created_at": datetime.now(),
                    "file_path": str(file_path),
                    "processing_mode": request.processing_mode,
                    "extension_type": "taxonomy_extension"
                }
            else:
                 raise HTTPException(status_code=500, detail="Failed to locate file path for taxonomy update")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extend taxonomy: {str(e)}")
