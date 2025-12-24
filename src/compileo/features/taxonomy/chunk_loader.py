"""
Chunk loading utilities for taxonomy extraction.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.compileo.storage.src.project.database_repositories import ChunkRepository
from src.compileo.features.extraction.exceptions import DatabaseConnectionError, FileStorageError
from src.compileo.features.extraction.error_logging import extraction_logger


class ChunkLoader:
    """
    Handles loading of project chunks from processed outputs.
    """

    def __init__(self, chunk_repo: ChunkRepository):
        """
        Initialize ChunkLoader.

        Args:
            chunk_repo: Repository for chunk data
        """
        self.repo = chunk_repo

    def load_project_chunks(self, project_id: int, max_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load chunks for a project from processed outputs.

        Args:
            project_id: Project ID
            max_chunks: Maximum number of chunks to load

        Returns:
            List of chunk data dictionaries

        Raises:
            DatabaseConnectionError: If database query fails
            FileStorageError: If file reading fails
        """
        chunks = []
        loaded_files = 0
        failed_files = 0

        try:
            # Get chunk files from processed outputs
            chunk_files = self.repo.get_project_chunks(project_id)
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to query chunk files for project {project_id}: {e}")

        for chunk_file_record in chunk_files:
            # Handle sqlite3.Row objects, tuples, and lists
            if hasattr(chunk_file_record, '__getitem__') and len(chunk_file_record) > 0:
                chunk_file_path_str = chunk_file_record[0]
                chunk_file_path = Path(chunk_file_path_str)
            else:
                extraction_logger.log_warning(
                    "Invalid chunk file record format",
                    "load_project_chunks",
                    context={"record": chunk_file_record, "record_type": type(chunk_file_record)},
                    project_id=project_id
                )
                continue

            try:
                if chunk_file_path.exists():
                    file_size = chunk_file_path.stat().st_size

                    if chunk_file_path_str.endswith('.json'):
                        try:
                            with open(chunk_file_path, 'r', encoding='utf-8') as f:
                                chunk_data = json.load(f)

                            if isinstance(chunk_data, list):
                                chunks.extend(chunk_data)
                            elif isinstance(chunk_data, dict) and 'chunks' in chunk_data:
                                chunks.extend(chunk_data['chunks'])
                            else:
                                # Single chunk
                                chunks.append(chunk_data)
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            raise FileStorageError(f"Failed to parse JSON file {chunk_file_path_str}: {e}", chunk_file_path_str, "json_parse")

                    elif chunk_file_path_str.endswith(('.txt', '.md')):
                        try:
                            with open(chunk_file_path, 'r', encoding='utf-8') as f:
                                chunk_text = f.read().strip()
                                if chunk_text:
                                    chunks.append({
                                        'id': os.path.basename(chunk_file_path_str),
                                        'text': chunk_text
                                    })
                        except UnicodeDecodeError as e:
                            raise FileStorageError(f"Failed to read text file {chunk_file_path_str}: {e}", chunk_file_path_str, "text_read")
                    else:
                        extraction_logger.log_warning(
                            f"Unsupported file type: {chunk_file_path_str}",
                            "load_project_chunks",
                            context={"file_path": chunk_file_path_str},
                            project_id=project_id
                        )
                        continue

                    loaded_files += 1

                    # Log large files for performance monitoring
                    if file_size > 10 * 1024 * 1024:  # 10MB
                        extraction_logger.log_performance_issue(
                            "load_project_chunks",
                            file_size / (1024 * 1024),  # Size in MB
                            5.0,  # 5MB threshold
                            context={"file_path": chunk_file_path},
                            project_id=project_id
                        )

                else:
                    extraction_logger.log_warning(
                        f"Chunk file not found: {chunk_file_path_str}",
                        "load_project_chunks",
                        context={"file_path": chunk_file_path_str},
                        project_id=project_id
                    )
                    continue

                # Apply max_chunks limit
                if max_chunks and len(chunks) >= max_chunks:
                    chunks = chunks[:max_chunks]
                    break

            except (FileStorageError, OSError) as e:
                # Re-raise file errors
                failed_files += 1
                extraction_logger.log_error(
                    e,
                    "load_project_chunks",
                    context={"chunk_file_path": chunk_file_path_str},
                    project_id=project_id
                )
                continue
            except Exception as e:
                failed_files += 1
                extraction_logger.log_error(
                    e,
                    "load_project_chunks",
                    context={"chunk_file_path": chunk_file_path_str, "error_type": type(e).__name__},
                    project_id=project_id
                )
                continue

        if failed_files > 0:
            extraction_logger.log_warning(
                f"Failed to load {failed_files} out of {len(chunk_files)} chunk files",
                "load_project_chunks",
                context={"loaded_files": loaded_files, "failed_files": failed_files},
                project_id=project_id
            )

        return chunks