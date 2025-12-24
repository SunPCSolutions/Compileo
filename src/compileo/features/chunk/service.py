"""
Chunk service for managing chunk operations.
"""

import os
import json
import logging
from typing import List, Optional
from ...storage.src.project.database_repositories import ChunkRepository

logger = logging.getLogger(__name__)

class ChunkService:
    def __init__(self, db_connection):
        self.chunk_repo = ChunkRepository(db_connection)

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk by ID from both database and filesystem.
        """
        # 1. Get chunk to find file path
        chunk = self.chunk_repo.get_chunk_by_id(chunk_id)
        if not chunk:
            logger.warning(f"Chunk {chunk_id} not found for deletion")
            return False
        
        # 2. Delete file
        file_path = chunk.get('file_path')
        
        # Fallback: Check metadata if file_path column is empty
        if not file_path:
            metadata = chunk.get('metadata')
            if metadata:
                try:
                    if isinstance(metadata, str):
                        metadata_dict = json.loads(metadata)
                    elif isinstance(metadata, dict):
                        metadata_dict = metadata
                    else:
                        metadata_dict = {}
                    
                    file_path = metadata_dict.get('file_path')
                except Exception as e:
                    logger.warning(f"Failed to parse metadata for chunk {chunk_id}: {e}")

        if file_path:
            # Handle both absolute and relative paths
            if not os.path.isabs(file_path):
                # Construct absolute path relative to workspace root/storage
                # Assuming FileManager.get_storage_base_path logic
                from ...storage.src.project.file_manager import FileManager
                storage_base = FileManager.get_storage_base_path()
                abs_file_path = storage_base / file_path
                file_path_to_delete = str(abs_file_path)
            else:
                file_path_to_delete = file_path

            if os.path.exists(file_path_to_delete):
                try:
                    os.remove(file_path_to_delete)
                    logger.info(f"Deleted chunk file: {file_path_to_delete}")
                except OSError as e:
                    logger.error(f"Error deleting chunk file {file_path_to_delete}: {e}")
            else:
                logger.warning(f"Chunk file not found at {file_path_to_delete}")
        else:
            logger.warning(f"Chunk record {chunk_id} has no file_path")

        # 3. Delete from Qdrant (Placeholder)
        # TODO: Implement Qdrant deletion when vector search is fully integrated

        # 4. Delete from DB
        try:
            return self.chunk_repo.delete_chunk(chunk_id)
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id} from database: {e}")
            # Even if DB deletion fails, we return False to indicate overall failure
            return False

    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document from both database and filesystem.
        """
        # 1. Get all chunks
        chunks = self.chunk_repo.get_chunks_by_document(document_id)
        
        # 2. Delete files
        from ...storage.src.project.file_manager import FileManager
        storage_base = FileManager.get_storage_base_path()
        
        for chunk in chunks:
            file_path = chunk.get('file_path')
            
            # Fallback: Check metadata if file_path column is empty
            if not file_path:
                metadata = chunk.get('metadata')
                if metadata:
                    try:
                        if isinstance(metadata, str):
                            metadata_dict = json.loads(metadata)
                        elif isinstance(metadata, dict):
                            metadata_dict = metadata
                        else:
                            metadata_dict = {}
                        
                        file_path = metadata_dict.get('file_path')
                    except Exception:
                        pass

            if file_path:
                # Handle both absolute and relative paths
                if not os.path.isabs(file_path):
                    abs_file_path = storage_base / file_path
                    file_path_to_delete = str(abs_file_path)
                else:
                    file_path_to_delete = file_path

                if os.path.exists(file_path_to_delete):
                    try:
                        os.remove(file_path_to_delete)
                        logger.info(f"Deleted chunk file: {file_path_to_delete}")
                    except OSError as e:
                        logger.error(f"Error deleting chunk file {file_path_to_delete}: {e}")
                else:
                    logger.warning(f"Chunk file not found at {file_path_to_delete}, proceeding with DB deletion")

        # 3. Delete from Qdrant (Placeholder)
        # TODO: Implement Qdrant deletion when vector search is fully integrated

        # 4. Delete from DB
        try:
            # ChunkRepository doesn't have delete_chunks_by_document, so we implement it manually
            deleted_count = 0
            for chunk in chunks:
                if self.chunk_repo.delete_chunk(chunk['id']):
                    deleted_count += 1
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id} from database: {e}")
            return 0

    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete multiple chunks by their IDs from both database and filesystem.
        Returns the number of successfully deleted chunks.
        """
        deleted_count = 0

        for chunk_id in chunk_ids:
            if self.delete_chunk(chunk_id):
                deleted_count += 1

        return deleted_count
