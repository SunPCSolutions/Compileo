"""
Chunk management API routes for the Compileo API.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from pydantic import BaseModel
from ...storage.src.database import get_db_connection
from ...features.chunk.service import ChunkService
from ...core.logging import get_logger

logger = get_logger(__name__)

class BatchDeleteRequest(BaseModel):
    chunk_ids: List[str]

# Create router
router = APIRouter()

# Dependency to get database connection
def get_db():
    return get_db_connection()

@router.delete("/batch")
async def delete_chunks_batch(request: BatchDeleteRequest, db=Depends(get_db)):
    """
    Delete multiple chunks by their IDs from both database and filesystem.
    """
    try:
        logger.debug(f"Batch delete requested for chunk_ids: {request.chunk_ids}")
        chunk_service = ChunkService(db)
        deleted_count = chunk_service.delete_chunks(request.chunk_ids)
        logger.debug(f"Batch delete completed, deleted_count: {deleted_count}")

        return {
            "message": f"Successfully deleted {deleted_count} chunks",
            "deleted_count": deleted_count
        }

    except Exception as e:
        logger.error(f"Batch delete error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete chunks: {str(e)}")

@router.delete("/document/{document_id}")
async def delete_document_chunks(document_id: str, db=Depends(get_db)):
    """
    Delete all chunks associated with a document from both database and filesystem.
    """
    try:
        chunk_service = ChunkService(db)
        deleted_count = chunk_service.delete_document_chunks(document_id)

        return {
            "message": f"Deleted {deleted_count} chunks for document {document_id}",
            "deleted_count": deleted_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document chunks: {str(e)}")

@router.delete("/{chunk_id}")
async def delete_chunk(chunk_id: str, db=Depends(get_db)):
    """
    Delete a specific chunk by ID from both database and filesystem.
    """
    try:
        chunk_service = ChunkService(db)
        success = chunk_service.delete_chunk(chunk_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Chunk with ID {chunk_id} not found")

        return {"message": f"Chunk {chunk_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chunk: {str(e)}")

@router.get("/document/{document_id}")
async def get_document_chunks(document_id: str, db=Depends(get_db)):
    """
    Retrieve all chunks for a specific document.
    """
    try:
        from ...features.extraction.chunk_loader import ChunkLoader
        from ...storage.src.project.database_repositories import ChunkRepository
        chunk_repo = ChunkRepository(db)
        chunk_loader = ChunkLoader(chunk_repo)
        chunks = chunk_loader.load_document_chunks(document_id)

        return {
            "document_id": document_id,
            "chunks": chunks,
            "total": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document chunks: {str(e)}")

@router.get("/project/{project_id}")
async def get_project_chunks(project_id: str, limit: int = 100, db=Depends(get_db)):
    """
    Retrieve chunks for all documents in a project.
    """
    try:
        from ...storage.src.project.database_repositories import ChunkRepository
        cursor = db.cursor()
        
        # We need more than just file_path for general info, but let's see what ChunkRepository has
        # and maybe add a more flexible query here
        cursor.execute('''
            SELECT c.*
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.project_id = ?
            ORDER BY d.created_at DESC, c.chunk_index ASC
            LIMIT ?
        ''', (project_id, limit))
        
        rows = cursor.fetchall()
        chunks = [dict(row) for row in rows] if rows else []

        return {
            "project_id": project_id,
            "chunks": chunks,
            "total": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve project chunks: {str(e)}")