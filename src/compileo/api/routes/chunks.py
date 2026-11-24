"""
Chunk management API routes for the Compileo API.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from pydantic import BaseModel
from ...storage.src.database import get_db_connection
from ..core.database import get_db
from ..core.dependencies import get_current_active_user
from ..models.user import User as UserSchema
from ...features.chunk.service import ChunkService

class BatchDeleteRequest(BaseModel):
    chunk_ids: List[str]

# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.delete("/batch")
async def delete_chunks_batch(
    request: BatchDeleteRequest,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Delete multiple chunks by their IDs from both database and filesystem for current user.
    """
    try:
        # Verify ownership of chunks first
        cursor = db.cursor()
        placeholders = ','.join(['?'] * len(request.chunk_ids))
        cursor.execute(f"SELECT COUNT(*) FROM chunks WHERE id IN ({placeholders}) AND owner_id = ?", (*request.chunk_ids, current_user.id))
        count = cursor.fetchone()[0]
        if count != len(request.chunk_ids):
             raise HTTPException(status_code=403, detail="Some chunks not found or access denied")

        logger.debug(f"DEBUG: Batch delete requested for chunk_ids: {request.chunk_ids}")
        chunk_service = ChunkService(current_user.id)
        deleted_count = chunk_service.delete_chunks(request.chunk_ids, owner_id=current_user.id)
        logger.debug(f"DEBUG: Batch delete completed, deleted_count: {deleted_count}")

        return {
            "message": f"Successfully deleted {deleted_count} chunks",
            "deleted_count": deleted_count
        }

    except Exception as e:
        import traceback
        logger.error(f"DEBUG: Batch delete error: {str(e)}")
        logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chunks: {str(e)}")

@router.delete("/document/{document_id}")
async def delete_document_chunks(
    document_id: str,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Delete all chunks associated with a document for current user from both database and filesystem.
    """
    try:
        # Verify ownership
        cursor = db.cursor()
        cursor.execute("SELECT id FROM documents WHERE id = ? AND owner_id = ?", (document_id, current_user.id))
        if not cursor.fetchone():
             raise HTTPException(status_code=404, detail="Document not found or access denied")

        chunk_service = ChunkService(current_user.id)
        deleted_count = chunk_service.delete_document_chunks(document_id, owner_id=current_user.id)

        return {
            "message": f"Deleted {deleted_count} chunks for document {document_id}",
            "deleted_count": deleted_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document chunks: {str(e)}")

@router.delete("/{chunk_id}")
async def delete_chunk(
    chunk_id: str,
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Delete a specific chunk by ID for current user from both database and filesystem.
    """
    try:
        # Verify ownership
        cursor = db.cursor()
        cursor.execute("SELECT id FROM chunks WHERE id = ? AND owner_id = ?", (chunk_id, current_user.id))
        if not cursor.fetchone():
             raise HTTPException(status_code=404, detail="Chunk not found or access denied")

        chunk_service = ChunkService(current_user.id)
        success = chunk_service.delete_chunk(chunk_id, owner_id=current_user.id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Chunk with ID {chunk_id} not found")

        return {"message": f"Chunk {chunk_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chunk: {str(e)}")

@router.get("/document/{document_id}")
async def get_document_chunks(
    document_id: str, 
    db=Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Retrieve all chunks for a specific document for current user.
    """
    try:
        from ...features.extraction.chunk_loader import ChunkLoader
        from ...storage.src.project.database_repositories import ChunkRepository
        
        # Verify ownership
        cursor = db.cursor()
        cursor.execute("SELECT id FROM documents WHERE id = ? AND owner_id = ?", (document_id, current_user.id))
        if not cursor.fetchone():
             raise HTTPException(status_code=404, detail="Document not found or access denied")

        # Use ChunkLoader to load full content from filesystem
        chunk_repo = ChunkRepository(current_user.id)
        chunk_loader = ChunkLoader(chunk_repo)
        chunks_with_content = chunk_loader.load_document_chunks(document_id, owner_id=current_user.id)
        
        # Format chunks for API response - map 'text' to 'content' for GUI compatibility
        formatted_chunks = []
        for chunk in chunks_with_content:
            text_content = chunk.get('text')
            formatted_chunk = {
                'id': chunk.get('id'),
                'content': text_content or chunk.get('content_preview', ''),  # Full content from filesystem
                'content_preview': (text_content or '')[:200],
                'chunk_index': chunk.get('chunk_index'),
                'document_id': chunk.get('document_id'),
                'metadata': chunk.get('metadata', {}),
                'created_at': chunk.get('created_at')
            }
            formatted_chunks.append(formatted_chunk)

        return {
            "document_id": document_id,
            "chunks": formatted_chunks,
            "total": len(formatted_chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document chunks: {str(e)}")