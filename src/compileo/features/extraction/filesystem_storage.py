"""
Filesystem storage implementation for extraction results.
Provides file-based storage with metadata management and integrity validation.
"""

import os
import json
import hashlib
import logging
import tempfile
import copy
import fcntl
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import jsonschema
from pydantic import BaseModel, Field

from .storage_monitor import get_metrics, get_logger, get_alert_manager

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class FileIntegrityError(StorageError):
    """Raised when file integrity validation fails."""
    pass


class StorageOperationError(StorageError):
    """Raised when a storage operation fails."""
    pass


class DirectoryCreationError(StorageError):
    """Raised when directory creation fails."""
    pass


class SchemaValidationError(StorageError):
    """Raised when data doesn't match the expected schema."""
    pass


class ExtractionResultMetadata(BaseModel):
    """Metadata for extraction result files."""
    job_id: str
    project_id: Union[int, str]
    taxonomy_id: Union[int, str]
    created_at: datetime
    updated_at: datetime
    checksum: str
    total_results: int
    avg_confidence: float
    categories_found: List[str]
    file_version: str = "1.0"
    chunk_size: Optional[int] = None
    has_more_chunks: bool = False
    next_chunk_file: Optional[str] = None


class ExtractionResultItem(BaseModel):
    """Individual extraction result item."""
    chunk_id: str
    chunk_text: str
    classifications: Dict[str, Any]
    confidence_score: float
    categories_matched: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResultFile(BaseModel):
    """Complete extraction result file structure."""
    metadata: ExtractionResultMetadata
    results: List[ExtractionResultItem]


# JSON Schema for validation
EXTRACTION_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "minLength": 1},
                "project_id": {"type": ["integer", "string"], "minimum": 1},
                "taxonomy_id": {"type": ["integer", "string"], "minLength": 1},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
                "checksum": {"type": "string", "minLength": 64, "maxLength": 64},
                "total_results": {"type": "integer", "minimum": 0},
                "avg_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "categories_found": {"type": "array", "items": {"type": "string"}},
                "file_version": {"type": "string"},
                "chunk_size": {"type": ["integer", "null"], "minimum": 1},
                "has_more_chunks": {"type": "boolean"},
                "next_chunk_file": {"type": ["string", "null"]}
            },
            "required": ["job_id", "project_id", "taxonomy_id", "created_at", "updated_at",
                        "checksum", "total_results", "avg_confidence", "categories_found"]
        },
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "minLength": 1},
                    "chunk_text": {"type": "string"},
                    "classifications": {"type": "object"},
                    "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "categories_matched": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"}
                },
                "required": ["chunk_id", "chunk_text", "classifications", "confidence_score", "categories_matched"]
            },
        }
    },
    "required": ["metadata", "results"]
}


class SchemaValidator:
    """Validates extraction result data against JSON schema."""

    @staticmethod
    def validate_data(data: Dict[str, Any]) -> None:
        """
        Validate data against the extraction result schema.

        Args:
            data: The data to validate

        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            jsonschema.validate(instance=data, schema=EXTRACTION_RESULT_SCHEMA)
        except jsonschema.ValidationError as e:
            raise SchemaValidationError(f"Schema validation failed: {e.message}")
        except Exception as e:
            raise SchemaValidationError(f"Schema validation error: {e}")

    @staticmethod
    def validate_file_data(data: Dict[str, Any]) -> ExtractionResultFile:
        """
        Validate and parse data into Pydantic model.

        Args:
            data: The data to validate and parse

        Returns:
            Validated ExtractionResultFile instance

        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            # First validate against JSON schema
            SchemaValidator.validate_data(data)

            # Then parse with Pydantic for additional validation
            # Create a copy for Pydantic validation to avoid datetime issues
            pydantic_data = data.copy()
            # Convert ISO strings back to datetime objects for Pydantic
            if 'metadata' in pydantic_data:
                for key in ['created_at', 'updated_at']:
                    if key in pydantic_data['metadata'] and isinstance(pydantic_data['metadata'][key], str):
                        pydantic_data['metadata'][key] = datetime.fromisoformat(pydantic_data['metadata'][key].replace('Z', '+00:00'))

            return ExtractionResultFile(**pydantic_data)
        except Exception as e:
            raise SchemaValidationError(f"Data validation failed: {e}")


class ExtractionFileManager:
    """
    Manages filesystem operations for extraction result files.
    Handles directory creation, file naming, and atomic operations.
    """

    def __init__(self, base_path: str = "storage/extract"):
        """
        Initialize the file manager.

        Args:
            base_path: Base directory for extraction files
        """
        self.base_path = Path(base_path)
        self._ensure_base_directory()

    def _ensure_base_directory(self) -> None:
        """Ensure the base extraction directory exists."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured extraction directory exists: {self.base_path}")
        except Exception as e:
            raise DirectoryCreationError(f"Failed to create extraction directory {self.base_path}: {e}")

    def get_project_directory(self, project_id: int) -> Path:
        """
        Get the directory path for a specific project.

        Args:
            project_id: The project ID

        Returns:
            Path to the project's directory
        """
        return self.base_path / str(project_id)

    def ensure_project_directory(self, project_id: int) -> Path:
        """
        Ensure the project directory exists and return its path.

        Args:
            project_id: The project ID

        Returns:
            Path to the project's directory
        """
        project_dir = self.get_project_directory(project_id)
        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            return project_dir
        except Exception as e:
            raise DirectoryCreationError(f"Failed to create project directory {project_dir}: {e}")

    def list_job_files(self, job_id: str) -> List[Path]:
        """
        List all files for a job across all project directories.
        
        Args:
            job_id: The extraction job ID
            
        Returns:
            List of file paths
        """
        all_files = []
        
        # Check all project directories for this job's files
        # Ideally list_job_files should take project_id for efficiency
        if self.base_path.exists():
            for project_dir in self.base_path.iterdir():
                if project_dir.is_dir() and project_dir.name.isdigit():
                    try:
                        all_files.extend(list(project_dir.glob(f"extraction_{job_id}_*.json")))
                    except Exception:
                        pass
                        
        return all_files

    def generate_filename(self, job_id: str, timestamp: Optional[datetime] = None) -> str:
        """
        Generate a filename for extraction results.

        Args:
            job_id: The extraction job ID
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Generated filename
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        # Ensure job_id is treated as a string
        return f"extraction_{str(job_id)}_{timestamp_str}.json"

    def calculate_checksum(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 checksum for data integrity.
        Automatically excludes checksum fields from calculation.

        Args:
            data: The data to checksum

        Returns:
            Hexadecimal checksum string
        """
        # Create a copy and remove checksum fields for calculation
        data_copy = copy.deepcopy(data)
        if 'metadata' in data_copy and 'checksum' in data_copy['metadata']:
            del data_copy['metadata']['checksum']

        # Custom JSON encoder to handle datetime objects
        def json_encoder(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        data_str = json.dumps(data_copy, sort_keys=True, separators=(',', ':'), default=json_encoder)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def validate_checksum(self, data: Dict[str, Any], expected_checksum: str) -> bool:
        """
        Validate data against expected checksum.

        Args:
            data: The data to validate
            expected_checksum: The expected checksum

        Returns:
            True if checksum matches, False otherwise
        """
        calculated = self.calculate_checksum(data)
        return calculated == expected_checksum

    def atomic_write(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Atomically write data to file with rollback capability.

        Args:
            file_path: Path to write to
            data: Data to write

        Raises:
            StorageOperationError: If write operation fails
        """
        # Create temporary file in same directory for atomic operation
        temp_fd = None
        temp_path = None

        try:
            # Create temporary file
            temp_fd, temp_path_str = tempfile.mkstemp(
                dir=file_path.parent,
                prefix=f"{file_path.stem}_",
                suffix=".tmp"
            )
            temp_path = Path(temp_path_str)

            # Write data to temporary file
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                # Custom JSON encoder to handle datetime objects
                def json_encoder(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

                json.dump(data, f, indent=2, ensure_ascii=False, default=json_encoder)
            temp_fd = None  # File descriptor is now closed

            # Atomic move to final location
            temp_path.replace(file_path)
            logger.debug(f"Atomically wrote file: {file_path}")

        except Exception as e:
            # Cleanup on failure
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except:
                    pass
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise StorageOperationError(f"Failed to write file {file_path}: {e}")

    def read_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Read and validate data from file.

        Args:
            file_path: Path to read from

        Returns:
            The data from the file

        Raises:
            FileIntegrityError: If file integrity check fails
            StorageOperationError: If read operation fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate checksum if present and appears valid (64 char hex)
            if ('metadata' in data and 'checksum' in data['metadata'] and
                data['metadata']['checksum'] and
                len(data['metadata']['checksum']) == 64 and
                all(c in '0123456789abcdefABCDEF' for c in data['metadata']['checksum'])):
                expected_checksum = data['metadata']['checksum']
                # Remove checksum from data for validation
                data_copy = copy.deepcopy(data)
                if 'metadata' in data_copy and 'checksum' in data_copy['metadata']:
                    del data_copy['metadata']['checksum']

                if not self.validate_checksum(data_copy, expected_checksum):
                    raise FileIntegrityError(f"Checksum validation failed for {file_path}")

            return data

        except json.JSONDecodeError as e:
            raise StorageOperationError(f"Invalid JSON in file {file_path}: {e}")
        except FileNotFoundError:
            raise StorageOperationError(f"File not found: {file_path}")
        except Exception as e:
            raise StorageOperationError(f"Failed to read file {file_path}: {e}")

    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file safely.

        Args:
            file_path: Path to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def list_project_files(self, project_id: int) -> List[Path]:
        """
        List all extraction files for a project.

        Args:
            project_id: The project ID

        Returns:
            List of file paths
        """
        project_dir = self.get_project_directory(project_id)
        if not project_dir.exists():
            return []

        try:
            return list(project_dir.glob("extraction_*.json"))
        except Exception as e:
            logger.error(f"Failed to list files for project {project_id}: {e}")
            return []


    def acquire_file_lock(self, file_path: Path, exclusive: bool = True) -> Optional[int]:
        """
        Acquire a file lock for concurrent access control.

        Args:
            file_path: Path to the file to lock
            exclusive: Whether to acquire an exclusive lock (write) or shared lock (read)

        Returns:
            File descriptor if lock acquired, None if failed
        """
        try:
            fd = os.open(file_path, os.O_RDONLY | os.O_CREAT)
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
            return fd
        except (OSError, IOError):
            # Failed to acquire lock
            return None

    def release_file_lock(self, fd: int) -> None:
        """
        Release a file lock.

        Args:
            fd: File descriptor from acquire_file_lock
        """
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except (OSError, IOError):
            pass  # Ignore errors when releasing


class FilesystemStorageManager:
    """
    Manages extraction result storage on the filesystem.
    Provides CRUD operations with integrity validation and atomic writes.
    """

    def __init__(self, file_manager: ExtractionFileManager):
        """
        Initialize the storage manager.

        Args:
            file_manager: The file manager instance
        """
        self.file_manager = file_manager

    def store_results(self, job_id: str, project_id: int, taxonomy_id: int,
                      results: List[Dict[str, Any]], chunk_size: Optional[int] = None) -> List[str]:
        """
        Store extraction results to filesystem, potentially in chunks.

        Args:
            job_id: The extraction job ID
            project_id: The project ID
            taxonomy_id: The taxonomy ID
            results: List of result dictionaries
            chunk_size: Optional chunk size for splitting large result sets

        Returns:
            List of stored file paths (relative to base directory)
        """
        start_time = time.time()
        success = False
        stored_files = []

        try:
            if not results:
                logger.warning(f"No extraction results to store for job {job_id}")
                return []

            # Calculate metadata
            total_results = len(results)
            avg_confidence = sum(r.get('confidence_score', 0.0) for r in results) / total_results

            # Collect all categories
            categories_found = set()
            for result in results:
                categories_found.update(result.get('categories_matched', []))

            categories_found = sorted(list(categories_found))

            # Ensure project directory exists
            project_dir = self.file_manager.ensure_project_directory(project_id)

            current_chunk = []
            chunk_index = 0

            # Process results in chunks if specified
            effective_chunk_size = chunk_size or len(results)

            for i, result in enumerate(results):
                current_chunk.append(result)

                # Check if we should write this chunk
                if len(current_chunk) >= effective_chunk_size or i == len(results) - 1:
                    # Create metadata for this chunk/file
                    now = datetime.utcnow()
                    has_more_chunks = (i < len(results) - 1) or (chunk_size and len(results) > chunk_size)

                    # Ensure taxonomy_id is string
                    metadata = ExtractionResultMetadata(
                        job_id=job_id,
                        project_id=project_id,
                        taxonomy_id=str(taxonomy_id),
                        created_at=now,
                        updated_at=now,
                        checksum="",  # Will be calculated after data preparation
                        total_results=len(current_chunk),
                        avg_confidence=sum(r.get('confidence_score', 0.0) for r in current_chunk) / len(current_chunk),
                        categories_found=categories_found,
                        chunk_size=chunk_size,
                        has_more_chunks=bool(has_more_chunks),
                        next_chunk_file=None
                    )

                    # Convert results to proper format
                    result_items = [
                        ExtractionResultItem(
                            chunk_id=r.get('chunk_id', ''),
                            chunk_text=r.get('chunk_text', ''),
                            classifications=r.get('classifications', {}),
                            confidence_score=r.get('confidence_score', 0.0),
                            categories_matched=r.get('categories_matched', []),
                            metadata=r.get('metadata', {})
                        ) for r in current_chunk
                    ]

                    # Prepare file data
                    file_data = ExtractionResultFile(
                        metadata=metadata,
                        results=result_items
                    )

                    # Convert to dict with proper JSON serialization
                    data_dict = file_data.dict()
                    # Ensure datetime objects are serialized as ISO strings
                    if 'metadata' in data_dict:
                        for key in ['created_at', 'updated_at']:
                            if key in data_dict['metadata'] and isinstance(data_dict['metadata'][key], datetime):
                                data_dict['metadata'][key] = data_dict['metadata'][key].isoformat()

                    # Calculate checksum (exclude checksum field from calculation)
                    checksum_data = data_dict.copy()
                    if 'metadata' in checksum_data and 'checksum' in checksum_data['metadata']:
                        del checksum_data['metadata']['checksum']
                    checksum = self.file_manager.calculate_checksum(checksum_data)
                    data_dict['metadata']['checksum'] = checksum

                    # Generate filename
                    filename = self.file_manager.generate_filename(job_id, now)
                    if chunk_index > 0:
                        # Add chunk suffix for additional chunks
                        name_parts = filename.rsplit('.', 1)
                        filename = f"{name_parts[0]}_chunk_{chunk_index}.{name_parts[1]}"

                    # Use project directory for file path
                    file_path = project_dir / filename

                    # Validate data before writing
                    SchemaValidator.validate_file_data(data_dict)

                    # Atomic write
                    self.file_manager.atomic_write(file_path, data_dict)

                    # Update next chunk file reference for previous chunk
                    if chunk_index > 0:
                        prev_file = stored_files[-1]
                        # Need to reconstruct full path for previous file update
                        # prev_file stored as relative path, so join with base_path
                        prev_file_path = self.file_manager.base_path / prev_file
                        self._update_next_chunk_reference(prev_file_path, filename)

                    stored_files.append(str(file_path.relative_to(self.file_manager.base_path)))
                    current_chunk = []
                    chunk_index += 1

            success = True
            logger.info(f"Stored {total_results} extraction results for job {job_id} in {len(stored_files)} files")
            return stored_files

        except Exception as e:
            get_logger().log_error('store_results', e, {'job_id': job_id, 'total_results': len(results) if 'results' in locals() else 0})
            raise
        finally:
            duration = time.time() - start_time
            get_metrics().record_operation('store_results', duration, success)
            get_logger().log_operation('store_results', job_id, duration, success,
                                     {'files_created': len(stored_files), 'total_results': len(results) if 'results' in locals() else 0})

    def _update_next_chunk_reference(self, file_path: Path, next_filename: str) -> None:
        """Update the next chunk file reference in a stored file."""
        try:
            data = self.file_manager.read_file(file_path)
            data['metadata']['next_chunk_file'] = next_filename
            self.file_manager.atomic_write(file_path, data)
        except Exception as e:
            logger.warning(f"Failed to update next chunk reference in {file_path}: {e}")

    def retrieve_results(self, job_id: str, page: int = 1, page_size: int = 50,
                        min_confidence: Optional[float] = None,
                        category_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], bool, int]:
        """
        Retrieve extraction results for a job with pagination and filtering.

        Args:
            job_id: The extraction job ID
            page: Page number (1-based)
            page_size: Number of results per page
            min_confidence: Minimum confidence threshold
            category_filter: Filter by category

        Returns:
            Tuple of (results, has_more, total_count)
        """
        start_time = time.time()
        success = False
        paginated_results = []

        try:
            job_files = self.file_manager.list_job_files(job_id)
            if not job_files:
                success = True
                return [], False, 0

            all_results = []
            total_count = 0

            # Read all files for this job
            for file_path in sorted(job_files):  # Sort to ensure consistent ordering
                try:
                    data = self.file_manager.read_file(file_path)
                    file_results = data.get('results', [])

                    # Apply filters
                    filtered_results = []
                    for result in file_results:
                        # Confidence filter
                        if min_confidence is not None and result.get('confidence_score', 0.0) < min_confidence:
                            continue

                        # Category filter
                        if category_filter and category_filter not in result.get('categories_matched', []):
                            continue

                        filtered_results.append(result)

                    all_results.extend(filtered_results)
                    total_count += len(filtered_results)

                except Exception as e:
                    logger.error(f"Failed to read results from {file_path}: {e}")
                    get_logger().log_error('retrieve_results_file_read', e, {'file_path': str(file_path), 'job_id': job_id})
                    continue

            # Apply pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_results = all_results[start_idx:end_idx]
            has_more = end_idx < len(all_results)

            success = True
            return paginated_results, has_more, total_count

        except Exception as e:
            get_logger().log_error('retrieve_results', e, {'job_id': job_id, 'page': page, 'page_size': page_size})
            raise
        finally:
            duration = time.time() - start_time
            get_metrics().record_operation('retrieve_results', duration, success)
            get_logger().log_operation('retrieve_results', job_id, duration, success,
                                     {'page': page, 'page_size': page_size, 'results_returned': len(paginated_results) if 'paginated_results' in locals() else 0})

    def get_job_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get aggregated metadata for all files of a job.

        Args:
            job_id: The extraction job ID

        Returns:
            Aggregated metadata or None if no files found
        """
        job_files = self.file_manager.list_job_files(job_id)
        if not job_files:
            return None

        total_results = 0
        all_categories = set()
        confidence_sum = 0.0
        earliest_created = None
        latest_updated = None

        for file_path in job_files:
            try:
                data = self.file_manager.read_file(file_path)
                metadata = data.get('metadata', {})

                total_results += metadata.get('total_results', 0)
                confidence_sum += metadata.get('avg_confidence', 0.0) * metadata.get('total_results', 0)
                all_categories.update(metadata.get('categories_found', []))

                created_at = metadata.get('created_at')
                updated_at = metadata.get('updated_at')

                if created_at:
                    created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if earliest_created is None or created_dt < earliest_created:
                        earliest_created = created_dt

                if updated_at:
                    updated_dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    if latest_updated is None or updated_dt > latest_updated:
                        latest_updated = updated_dt

            except Exception as e:
                logger.error(f"Failed to read metadata from {file_path}: {e}")
                continue

        if total_results == 0:
            return None

        avg_confidence = confidence_sum / total_results

        return {
            'job_id': job_id,
            'total_results': total_results,
            'avg_confidence': avg_confidence,
            'categories': sorted(list(all_categories)),
            'created_at': earliest_created.isoformat() if earliest_created else None,
            'updated_at': latest_updated.isoformat() if latest_updated else None,
            'file_count': len(job_files)
        }

    def delete_job_results(self, job_id: str) -> bool:
        """
        Delete all result files for a job.

        Args:
            job_id: The extraction job ID

        Returns:
            True if all files deleted successfully
        """
        job_files = self.file_manager.list_job_files(job_id)
        success = True

        for file_path in job_files:
            if not self.file_manager.delete_file(file_path):
                success = False
                logger.error(f"Failed to delete file: {file_path}")

        # Try to clean up any empty project directories that might contain this job's files
        # This is more aggressive cleanup for project-based storage
        if self.file_manager.base_path.exists():
            for project_dir in self.file_manager.base_path.iterdir():
                if project_dir.is_dir() and project_dir.name.isdigit():
                    try:
                        # Check if this project directory has any files for this job
                        job_files_in_project = list(project_dir.glob(f"extraction_{job_id}_*.json"))
                        if not job_files_in_project:
                            # No files for this job, check if directory is empty
                            if not list(project_dir.iterdir()):
                                project_dir.rmdir()
                                logger.debug(f"Removed empty project directory: {project_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup project directory {project_dir}: {e}")

        return success

    def atomic_multi_file_operation(self, operations: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """
        Perform atomic multi-file operations with rollback capability.

        Args:
            operations: List of (operation_type, data) tuples
                       operation_type can be 'store' or 'delete'
                       For 'store': data contains file_path and content
                       For 'delete': data contains file_path

        Returns:
            List of successfully completed operations

        Raises:
            StorageOperationError: If any operation fails (rollback attempted)
        """
        completed_operations = []
        created_files = []

        try:
            for operation_type, data in operations:
                if operation_type == 'store':
                    file_path = Path(data['file_path'])
                    content = data['content']

                    # Store the file
                    self.file_manager.atomic_write(file_path, content)
                    created_files.append(file_path)
                    completed_operations.append(f"store:{file_path}")

                elif operation_type == 'delete':
                    file_path = Path(data['file_path'])

                    # Delete the file
                    if self.file_manager.delete_file(file_path):
                        completed_operations.append(f"delete:{file_path}")
                    else:
                        logger.warning(f"File {file_path} not found for deletion")

                else:
                    raise StorageOperationError(f"Unknown operation type: {operation_type}")

        except Exception as e:
            # Rollback: delete any files we created
            logger.warning(f"Operation failed, rolling back {len(created_files)} files")
            for file_path in created_files:
                try:
                    if file_path.exists():
                        file_path.unlink()
                        logger.debug(f"Rolled back file: {file_path}")
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback file {file_path}: {rollback_error}")

            raise StorageOperationError(f"Multi-file operation failed and was rolled back: {e}")

        return completed_operations

    def check_file_integrity(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Check integrity of stored files.

        Args:
            job_id: Specific job ID to check, or None for all jobs

        Returns:
            Integrity check results
        """
        start_time = time.time()
        success = False

        results = {
            'total_files': 0,
            'valid_files': 0,
            'corrupted_files': 0,
            'missing_checksums': 0,
            'corrupted_files_list': [],
            'checked_at': datetime.utcnow().isoformat()
        }

        try:
            if job_id:
                job_files = self.file_manager.list_job_files(job_id)
            else:
                # Check all directories (legacy job_ and new project dirs)
                job_files = []
                if self.file_manager.base_path.exists():
                    for directory in self.file_manager.base_path.iterdir():
                        if directory.is_dir():
                            # Check if it's a project dir
                            if directory.name.isdigit():
                                job_files.extend(directory.glob("extraction_*.json"))

            for file_path in job_files:
                results['total_files'] += 1
                try:
                    data = self.file_manager.read_file(file_path)
                    # If we get here, the file is valid (checksum passed)
                    results['valid_files'] += 1
                except FileIntegrityError:
                    results['corrupted_files'] += 1
                    results['corrupted_files_list'].append(str(file_path.relative_to(self.file_manager.base_path)))
                    get_alert_manager().trigger_alert(
                        'integrity',
                        'error',
                        f"File integrity check failed: {file_path.name}",
                        {'file_path': str(file_path), 'job_id': job_id}
                    )
                except Exception as e:
                    # Other errors (JSON parsing, etc.)
                    results['corrupted_files'] += 1
                    results['corrupted_files_list'].append(str(file_path.relative_to(self.file_manager.base_path)))
                    logger.warning(f"Error checking file {file_path}: {e}")
                    get_logger().log_error('integrity_check_file', e, {'file_path': str(file_path)})

            success = True

        except Exception as e:
            logger.error(f"Error during integrity check: {e}")
            results['error'] = str(e)
            get_logger().log_error('integrity_check', e, {'job_id': job_id})
        finally:
            duration = time.time() - start_time
            get_metrics().record_operation('check_file_integrity', duration, success)
            get_logger().log_operation('check_file_integrity', job_id, duration, success, results)

        return results


class HybridStorageManager:
    """
    Hybrid storage manager that combines database metadata with filesystem data.
    Provides a unified interface for extraction result storage and retrieval.
    """

    def __init__(self, filesystem_manager: FilesystemStorageManager,
                 database_repo=None, enable_filesystem: bool = True):
        """
        Initialize the hybrid storage manager.

        Args:
            filesystem_manager: The filesystem storage manager
            database_repo: Optional database repository for metadata
            enable_filesystem: Whether to use filesystem storage
        """
        self.filesystem_manager = filesystem_manager
        self.database_repo = database_repo
        self.enable_filesystem = enable_filesystem

    def store_results(self, job_id: str, project_id: int, taxonomy_id: int,
                      results: List[Dict[str, Any]], chunk_size: Optional[int] = None,
                      storage_preference: str = "filesystem") -> Dict[str, Any]:
        """
        Store extraction results using hybrid approach.

        Args:
            job_id: The extraction job ID
            project_id: The project ID
            taxonomy_id: The taxonomy ID
            results: List of result dictionaries
            chunk_size: Optional chunk size for splitting large result sets
            storage_preference: Preferred storage method ('filesystem', 'database', 'hybrid')

        Returns:
            Storage result information
        """
        stored_files = []
        db_success = False

        # Store in filesystem if enabled and preferred
        if self.enable_filesystem and storage_preference in ['filesystem', 'hybrid']:
            try:
                stored_files = self.filesystem_manager.store_results(
                    job_id, project_id, taxonomy_id, results, chunk_size
                )
                if results:
                    logger.info(f"Stored {len(results)} results in filesystem for job {job_id}")
                else:
                    logger.info(f"No results to store in filesystem for job {job_id}")
            except Exception as e:
                logger.error(f"Filesystem storage failed for job {job_id}: {e}")
                if storage_preference == 'filesystem':
                    raise

        # Store metadata in database if available (only for hybrid mode)
        if self.database_repo and storage_preference == 'hybrid':
            try:
                # Store only metadata in database for indexing/querying
                # Calculate file path based on job files - this is an approximation as results may span multiple files
                # For exact mapping we would need to know which file each result went into, but stored_files helps
                primary_file_path = stored_files[0] if stored_files else None
                
                for result in results:
                    self.database_repo.create_result(
                        job_id=job_id,
                        chunk_id=result.get('chunk_id', ''),
                        categories=result.get('categories_matched', []),
                        confidence=result.get('confidence_score', 0.0),
                        project_id=project_id,
                        extracted_data={}, # Empty dict to avoid redundancy with filesystem data
                        file_path=primary_file_path
                    )
                db_success = True
                if results:
                    logger.info(f"Stored {len(results)} result metadata in database for job {job_id}")
                else:
                    logger.info(f"No result metadata to store in database for job {job_id}")
            except Exception as e:
                logger.error(f"Database metadata storage failed for job {job_id}: {e}")
                if storage_preference == 'database':
                    raise

        return {
            'job_id': job_id,
            'total_results': len(results),
            'filesystem_files': stored_files,
            'database_stored': db_success,
            'storage_preference': storage_preference
        }

    def retrieve_results(self, job_id: str, page: int = 1, page_size: int = 50,
                        min_confidence: Optional[float] = None,
                        category_filter: Optional[str] = None,
                        storage_preference: str = "filesystem") -> Tuple[List[Dict[str, Any]], bool, int]:
        """
        Retrieve extraction results using hybrid approach.

        Args:
            job_id: The extraction job ID
            page: Page number (1-based)
            page_size: Number of results per page
            min_confidence: Minimum confidence threshold
            category_filter: Filter by category
            storage_preference: Preferred retrieval method

        Returns:
            Tuple of (results, has_more, total_count)
        """
        # Try filesystem first if enabled
        if self.enable_filesystem and storage_preference in ['filesystem', 'hybrid']:
            try:
                results, has_more, total = self.filesystem_manager.retrieve_results(
                    job_id, page, page_size, min_confidence, category_filter
                )
                if results or storage_preference == 'filesystem':
                    return results, has_more, total
            except Exception as e:
                logger.warning(f"Filesystem retrieval failed for job {job_id}: {e}")
                if storage_preference == 'filesystem':
                    raise

        # Fallback to database if available
        if self.database_repo and storage_preference in ['database', 'hybrid']:
            try:
                results, has_more = self.database_repo.get_results_by_job(
                    job_id, page, page_size, min_confidence, category_filter
                )

                # Convert database format to consistent format
                converted_results = []
                for row in results:
                    extracted_data = json.loads(row['extracted_data']) if isinstance(row['extracted_data'], str) else row['extracted_data']
                    converted_results.append({
                        'chunk_id': row['chunk_id'],
                        'chunk_text': extracted_data.get('chunk_text', ''),
                        'classifications': extracted_data.get('classifications', {}),
                        'confidence_score': row['confidence'],
                        'categories_matched': json.loads(row['categories']) if isinstance(row['categories'], str) else row['categories'],
                        'metadata': extracted_data.get('metadata', {})
                    })

                # Get total count
                metadata = self.database_repo.get_result_metadata(job_id)
                total_count = metadata.get('total_results', 0)

                return converted_results, has_more, total_count

            except Exception as e:
                logger.error(f"Database retrieval failed for job {job_id}: {e}")
                if storage_preference == 'database':
                    raise

        return [], False, 0

    def get_job_metadata(self, job_id: str, storage_preference: str = "filesystem") -> Optional[Dict[str, Any]]:
        """
        Get job metadata using hybrid approach.

        Args:
            job_id: The extraction job ID
            storage_preference: Preferred metadata source

        Returns:
            Job metadata or None if not found
        """
        # Try filesystem first if enabled
        if self.enable_filesystem and storage_preference in ['filesystem', 'hybrid']:
            try:
                metadata = self.filesystem_manager.get_job_metadata(job_id)
                if metadata or storage_preference == 'filesystem':
                    return metadata
            except Exception as e:
                logger.warning(f"Filesystem metadata retrieval failed for job {job_id}: {e}")
                if storage_preference == 'filesystem':
                    raise

        # Fallback to database if available
        if self.database_repo and storage_preference in ['database', 'hybrid']:
            try:
                return self.database_repo.get_result_metadata(job_id)
            except Exception as e:
                logger.error(f"Database metadata retrieval failed for job {job_id}: {e}")
                if storage_preference == 'database':
                    raise

        return None

    def delete_job_results(self, job_id: str, storage_preference: str = "filesystem") -> Dict[str, Any]:
        """
        Delete job results using hybrid approach.

        Args:
            job_id: The extraction job ID
            storage_preference: What to delete

        Returns:
            Deletion result information
        """
        filesystem_success = False
        database_success = False

        # Delete from filesystem if enabled
        if self.enable_filesystem and storage_preference in ['filesystem', 'hybrid']:
            try:
                filesystem_success = self.filesystem_manager.delete_job_results(job_id)
                logger.info(f"Deleted filesystem results for job {job_id}: {filesystem_success}")
            except Exception as e:
                logger.error(f"Filesystem deletion failed for job {job_id}: {e}")
                if storage_preference == 'filesystem':
                    raise

        # Delete from database if available
        if self.database_repo and storage_preference in ['database', 'hybrid']:
            try:
                deleted_count = self.database_repo.delete_results_by_job(job_id)
                database_success = deleted_count > 0
                logger.info(f"Deleted {deleted_count} database results for job {job_id}")
            except Exception as e:
                logger.error(f"Database deletion failed for job {job_id}: {e}")
                if storage_preference == 'database':
                    raise

        return {
            'job_id': job_id,
            'filesystem_deleted': filesystem_success,
            'database_deleted': database_success,
            'storage_preference': storage_preference
        }