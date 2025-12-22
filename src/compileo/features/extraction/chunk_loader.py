"""
Chunk Loader Service

Handles loading and processing of document chunks for extraction operations.
"""

import json
import os
from typing import List, Dict, Any, Optional, Union
from src.compileo.storage.src.project.database_repositories import ChunkRepository


class ChunkLoader:
    """
    Service for loading and processing document chunks.

    Provides utilities for:
    - Loading chunks from database
    - Filtering and sorting chunks
    - Processing chunk metadata
    """

    def __init__(self, chunk_repo: ChunkRepository):
        """
        Initialize ChunkLoader.

        Args:
            chunk_repo: Repository for accessing chunk data
        """
        self.chunk_repo = chunk_repo

    def load_project_chunks(self, project_id: Union[str, int], max_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load chunks for a specific project.

        Args:
            project_id: Project ID to load chunks for
            max_chunks: Maximum number of chunks to load (None for all)

        Returns:
            List of chunk data dictionaries with keys: 'id', 'text', 'metadata', etc.

        Raises:
            DatabaseConnectionError: If database query fails
        """
        try:
            # First try loading directly from chunks table if content is stored there
            cursor = self.chunk_repo.cursor()
            cursor.execute('''
                SELECT c.id, c.content_preview, c.file_path, c.metadata
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.project_id = ?
                ORDER BY d.created_at DESC, c.chunk_index ASC
            ''', (project_id,))
            
            processed_chunks = []
            rows = cursor.fetchall()
            
            for row in rows:
                chunk_id, content_preview, file_path, metadata_str = row

                content = None

                # Always try to load full content from filesystem first
                if file_path:
                    try:
                        # Check direct path first, then try resolving relative to storage root
                        resolved_path = file_path
                        if not os.path.exists(resolved_path) and not file_path.startswith('storage/'):
                            # Try prefixing with storage/
                            potential_path = os.path.join('storage', file_path)
                            if os.path.exists(potential_path):
                                resolved_path = potential_path

                        if os.path.exists(resolved_path):
                            with open(resolved_path, 'r', encoding='utf-8') as f:
                                full_content = f.read().strip()
                                if full_content:
                                    content = full_content
                    except Exception:
                        # Fall back to preview if file reading fails
                        pass

                # Always include chunk, even if content load failed
                chunk_data = {
                    'id': chunk_id,
                    'text': content,  # May be None if load failed
                    'metadata': json.loads(metadata_str) if metadata_str else {}
                }
                processed_chunks.append(chunk_data)

                if max_chunks and len(processed_chunks) >= max_chunks:
                    break
            
            return processed_chunks

        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Failed to load chunks for project {project_id}: {e}") from e

    def load_document_chunks(self, document_id: Union[str, int], max_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load chunks for a specific document.

        Args:
            document_id: Document ID to load chunks for
            max_chunks: Maximum number of chunks to load (None for all)

        Returns:
            List of chunk data dictionaries
        """
        try:
            chunks = self.chunk_repo.get_chunks_by_document(document_id)

            processed_chunks = []
            for chunk in chunks:
                # load content from filesystem first
                content = None
                file_path = chunk.get('file_path')
                if file_path:
                    try:
                        # Check direct path first, then try resolving relative to storage root
                        resolved_path = file_path
                        if not os.path.exists(resolved_path) and not file_path.startswith('storage/'):
                            # Try prefixing with storage/
                            potential_path = os.path.join('storage', file_path)
                            if os.path.exists(potential_path):
                                resolved_path = potential_path

                        if os.path.exists(resolved_path):
                            with open(resolved_path, 'r', encoding='utf-8') as f:
                                full_content = f.read().strip()
                                if full_content:
                                    content = full_content
                    except Exception:
                        # Fall back to database preview if file reading fails
                        pass

                # Always include chunk, even if content load failed
                processed_chunk = {
                    'id': chunk.get('id'),
                    'text': content,  # May be None if load failed
                    'metadata': json.loads(chunk.get('metadata', '{}')) if chunk.get('metadata') else {},
                    'document_id': chunk.get('document_id'),
                    'chunk_index': chunk.get('chunk_index'),
                    'created_at': chunk.get('created_at')
                }
                processed_chunks.append(processed_chunk)

                if max_chunks and len(processed_chunks) >= max_chunks:
                    break

            return processed_chunks

        except Exception as e:
            raise Exception(f"Failed to load chunks for document {document_id}: {e}") from e


    def validate_chunk_data(self, chunk_data: Dict[str, Any]) -> bool:
        """
        Validate chunk data structure.

        Args:
            chunk_data: Chunk data to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['id', 'text']
        for field in required_fields:
            if field not in chunk_data:
                return False
            if field == 'text' and (not isinstance(chunk_data[field], str) or not chunk_data[field].strip()):
                return False
            if field == 'id' and chunk_data[field] is None:
                return False

        return True