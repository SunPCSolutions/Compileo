from typing import List, Dict, Any, Optional, Tuple, Callable
import threading
import json
import os
import glob
import sqlite3

from src.compileo.core.logging import get_logger
from src.compileo.storage.src.project.database_repositories import (
    ExtractionResultRepository, ExtractionJobRepository
)
from src.compileo.features.taxonomy.loader import TaxonomyLoader
from src.compileo.features.extraction.filesystem_storage import (
    FilesystemStorageManager, ExtractionFileManager
)

logger = get_logger(__name__)

class DataLoader:
    """
    Handles loading of data from extraction results and chunk files.
    """

    def __init__(
        self,
        extraction_result_repository: Optional[ExtractionResultRepository] = None,
        extraction_job_repository: Optional[ExtractionJobRepository] = None,
        taxonomy_loader: Optional[TaxonomyLoader] = None,
        filesystem_storage_manager: Optional[FilesystemStorageManager] = None,
        enable_filesystem_storage: bool = True
    ):
        self.extraction_result_repository = extraction_result_repository
        self.extraction_job_repository = extraction_job_repository
        self.taxonomy_loader = taxonomy_loader
        self.filesystem_storage_manager = filesystem_storage_manager
        self.enable_filesystem_storage = enable_filesystem_storage
        
        # Thread lock for file operations to prevent concurrent file descriptor issues
        self._file_lock = threading.Lock()

        # Initialize filesystem storage if not provided but enabled
        if self.enable_filesystem_storage and not self.filesystem_storage_manager:
            file_manager = ExtractionFileManager()
            self.filesystem_storage_manager = FilesystemStorageManager(file_manager)

    def load_extracted_data(self, project_id: int, min_confidence: float = 0.0,
                            use_filesystem: bool = True, load_chunk_text: bool = True,
                            only_validated: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Load extracted data for a project from filesystem (preferred) or database (fallback).
        """
        extracted_data = []

        # Try filesystem storage first if enabled
        if use_filesystem and self.enable_filesystem_storage and self.filesystem_storage_manager:
            try:
                logger.info(f"Attempting to load extracted data from filesystem for project {project_id}")

                # Get all completed jobs for this project to find result files
                if self.extraction_job_repository:
                    jobs = self.extraction_job_repository.get_jobs_by_project(project_id, status='completed')
                    job_ids = [job['id'] for job in jobs]
                else:
                    # No fallback for legacy structures
                    logger.info(f"No extraction job repository available for project {project_id}")
                    job_ids = []

                if not job_ids:
                    logger.info(f"No completed jobs found for project {project_id}")
                else:
                    # Load results from each job's files
                    for job_id in job_ids:
                        try:
                            results, _, _ = self.filesystem_storage_manager.retrieve_results(
                                job_id=job_id,
                                page=1,
                                page_size=10000,  # Large page size to get all results
                                min_confidence=min_confidence
                            )

                            for result in results:
                                # Apply validation filter if requested
                                if only_validated and not result.get('metadata', {}).get('is_validated', False):
                                    continue

                                # Convert filesystem format to standardized format
                                chunk_id = result.get('chunk_id', '')
                                chunk_text = result.get('chunk_text', '')
                                # If chunk_text is missing and we need it, load from chunk file
                                if not chunk_text and chunk_id and load_chunk_text:
                                    chunk_text = self.load_chunk_text(project_id, chunk_id)

                                # Extract entities from metadata.extracted_entities
                                entities = []
                                metadata = result.get('metadata', {})
                                if 'extracted_entities' in metadata:
                                    for category_data in metadata['extracted_entities'].values():
                                        if 'entities' in category_data:
                                            entities.extend(category_data['entities'])
                                        if 'texts' in category_data:
                                            entities.extend(category_data['texts'])

                                item = {
                                    'chunk_id': chunk_id,
                                    'categories': result.get('categories_matched', []),
                                    'confidence': result.get('confidence_score', 0.0),
                                    'chunk_text': chunk_text,
                                    'classifications': result.get('classifications', {}),
                                    'metadata': metadata,
                                    'entities': entities,  # Add extracted entities
                                    'source_type': 'extraction'
                                }
                                extracted_data.append(item)

                        except Exception as e:
                            logger.warning(f"Failed to load results for job {job_id}: {e}")
                            continue

                if extracted_data:
                    logger.info(f"Loaded {len(extracted_data)} extraction results from filesystem for project {project_id}")
                    return extracted_data

            except Exception as e:
                logger.warning(f"Filesystem loading failed for project {project_id}: {e}")

        # Fallback to database if filesystem failed or disabled
        if self.extraction_result_repository:
            try:
                logger.info(f"Falling back to database loading for project {project_id}")

                # Get all extraction results for the project
                if hasattr(self.extraction_result_repository, 'get_results_by_project'):
                    results = self.extraction_result_repository.get_results_by_project(
                        project_id=project_id,
                        min_confidence=min_confidence
                    )
                else:
                    logger.warning("ExtractionResultRepository missing get_results_by_project method")
                    results = []

                if not results:
                    logger.info(f"No extraction results found for project {project_id}")
                    return None

                # Convert database results to usable format
                for result in results:
                    # Parse JSON fields - result is a sqlite3.Row object
                    try:
                        categories = json.loads(result['categories']) if result['categories'] else []
                        confidence = float(result['confidence']) if result['confidence'] is not None else 0.0
                        extracted_data_dict = json.loads(result['extracted_data']) if result['extracted_data'] else {}
                        
                        # Apply validation filter if requested
                        if only_validated and not extracted_data_dict.get('is_validated', False):
                            continue

                        # Create a standardized format for dataset generation
                        chunk_id = str(result['chunk_id']) if result['chunk_id'] else ''
                        chunk_text = extracted_data_dict.get('chunk_text', '')
                        # If chunk_text is missing and we need it, load from chunk file
                        if not chunk_text and chunk_id and load_chunk_text:
                            chunk_text = self.load_chunk_text(project_id, chunk_id)

                        # Extract entities from metadata.extracted_entities
                        entities = []
                        if 'extracted_entities' in extracted_data_dict.get('metadata', {}):
                            for category_data in extracted_data_dict['metadata']['extracted_entities'].values():
                                if 'entities' in category_data:
                                    entities.extend(category_data['entities'])
                                if 'texts' in category_data:
                                    entities.extend(category_data['texts'])

                        item = {
                            'chunk_id': chunk_id,
                            'categories': categories,
                            'confidence': confidence,
                            'chunk_text': chunk_text,
                            'classifications': extracted_data_dict.get('classifications', {}),
                            'metadata': extracted_data_dict.get('metadata', {}),
                            'entities': entities,  # Add extracted entities
                            'source_type': 'extraction'
                        }
                        extracted_data.append(item)
                    except (json.JSONDecodeError, IndexError, TypeError) as e:
                        logger.warning(f"Failed to parse extraction result {result}: {e}")
                        continue

                logger.info(f"Loaded {len(extracted_data)} extraction results from database for project {project_id}")
                return extracted_data

            except Exception as e:
                logger.error(f"Failed to load extracted data from database for project {project_id}: {e}")
                return None

        logger.warning("No extraction result repository available, cannot load extracted data")
        return None

    def load_extracted_data_batch(self, project_id: int, min_confidence: float = 0.0,
                                  category_filter: Optional[str] = None,
                                  limit: Optional[int] = None, offset: Optional[int] = 0,
                                  use_filesystem: bool = True, load_chunk_text: bool = True,
                                  only_validated: bool = False,
                                  progress_callback: Optional[callable] = None) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Load extracted data in batches with pagination and filtering.
        """
        all_results = []
        has_more = False

        # Try filesystem storage first if enabled
        if use_filesystem and self.enable_filesystem_storage and self.filesystem_storage_manager:
            try:
                if progress_callback:
                    progress_callback(0.1, "Finding completed jobs...")

                # Get all completed jobs for this project
                if self.extraction_job_repository:
                    jobs = self.extraction_job_repository.get_jobs_by_project(project_id, status='completed')
                    job_ids = [job['id'] for job in jobs]
                else:
                    # No fallback for legacy structures
                    job_ids = []

                if not job_ids:
                    if progress_callback:
                        progress_callback(1.0, "No completed jobs found")
                    return [], False

                total_jobs = len(job_ids)
                results_collected = 0

                for i, job_id in enumerate(job_ids):
                    if progress_callback:
                        progress_callback(0.1 + (i / total_jobs) * 0.8, f"Processing job {job_id}...")

                    try:
                        # Get results with filtering
                        page = 1
                        page_size = min(limit or 1000, 1000)  # Reasonable page size

                        while True:
                            results, more, total = self.filesystem_storage_manager.retrieve_results(
                                job_id=job_id,
                                page=page,
                                page_size=page_size,
                                min_confidence=min_confidence,
                                category_filter=category_filter
                            )

                            # Convert to standardized format
                            for result in results:
                                # Apply validation filter if requested
                                if only_validated and not result.get('metadata', {}).get('is_validated', False):
                                    continue

                                if limit and len(all_results) >= limit:
                                    has_more = True
                                    break

                                chunk_id = result.get('chunk_id', '')
                                chunk_text = result.get('chunk_text', '')
                                # If chunk_text is missing and we need it, load from chunk file
                                if not chunk_text and chunk_id and load_chunk_text:
                                    chunk_text = self.load_chunk_text(project_id, chunk_id)

                                # Extract entities from metadata.extracted_entities
                                entities = []
                                metadata = result.get('metadata', {})
                                if 'extracted_entities' in metadata:
                                    for category_data in metadata['extracted_entities'].values():
                                        if 'entities' in category_data:
                                            entities.extend(category_data['entities'])
                                        if 'texts' in category_data:
                                            entities.extend(category_data['texts'])

                                item = {
                                    'chunk_id': chunk_id,
                                    'categories': result.get('categories_matched', []),
                                    'confidence': result.get('confidence_score', 0.0),
                                    'chunk_text': chunk_text,
                                    'classifications': result.get('classifications', {}),
                                    'metadata': metadata,
                                    'entities': entities,  # Add extracted entities
                                    'source_type': 'extraction'
                                }
                                all_results.append(item)
                                results_collected += 1

                            if not more or (limit and len(all_results) >= limit):
                                break
                            page += 1

                        if limit and len(all_results) >= limit:
                            has_more = True
                            break

                    except Exception as e:
                        logger.warning(f"Failed to load results for job {job_id}: {e}")
                        continue

                if progress_callback:
                    progress_callback(1.0, f"Loaded {len(all_results)} results")

                # Apply offset and limit
                if offset:
                    all_results = all_results[offset:]
                if limit:
                    has_more = has_more or len(all_results) > limit
                    all_results = all_results[:limit]

                return all_results, has_more

            except Exception as e:
                logger.warning(f"Filesystem batch loading failed for project {project_id}: {e}")

        # Fallback to database
        if self.extraction_result_repository:
            try:
                if progress_callback:
                    progress_callback(0.5, "Loading from database...")

                results = self.extraction_result_repository.get_results_by_project(
                    project_id=project_id,
                    min_confidence=min_confidence,
                    limit=limit,
                    offset=offset
                )

                # Convert database format to standardized format
                converted_results = []
                for result in results:
                    try:
                        categories = json.loads(result['categories']) if result['categories'] else []
                        confidence = float(result['confidence']) if result['confidence'] is not None else 0.0
                        extracted_data_dict = json.loads(result['extracted_data']) if result['extracted_data'] else {}

                        # Apply validation filter if requested
                        if only_validated and not extracted_data_dict.get('is_validated', False):
                            continue

                        # Apply category filter if specified
                        if category_filter and category_filter not in categories:
                            continue

                        chunk_id = str(result['chunk_id']) if result['chunk_id'] else ''
                        chunk_text = extracted_data_dict.get('chunk_text', '')
                        # If chunk_text is missing and we need it, load from chunk file
                        if not chunk_text and chunk_id and load_chunk_text:
                            chunk_text = self.load_chunk_text(project_id, chunk_id)

                        # Extract entities from metadata.extracted_entities
                        entities = []
                        if 'extracted_entities' in extracted_data_dict.get('metadata', {}):
                            for category_data in extracted_data_dict['metadata']['extracted_entities'].values():
                                if 'entities' in category_data:
                                    entities.extend(category_data['entities'])
                                if 'texts' in category_data:
                                    entities.extend(category_data['texts'])

                        item = {
                            'chunk_id': chunk_id,
                            'categories': categories,
                            'confidence': confidence,
                            'chunk_text': chunk_text,
                            'classifications': extracted_data_dict.get('classifications', {}),
                            'metadata': extracted_data_dict.get('metadata', {}),
                            'entities': entities,  # Add extracted entities
                            'source_type': 'extraction'
                        }
                        converted_results.append(item)
                    except (json.JSONDecodeError, IndexError, TypeError) as e:
                        logger.warning(f"Failed to parse extraction result {result}: {e}")
                        continue

                if progress_callback:
                    progress_callback(1.0, f"Loaded {len(converted_results)} results from database")

                # Check if there are more results (rough estimate)
                has_more = limit is not None and len(converted_results) >= limit
                return converted_results, has_more

            except Exception as e:
                logger.error(f"Database batch loading failed for project {project_id}: {e}")

        return [], False

    def load_extracted_data_by_job_id(self, project_id: int, extraction_job_id: str, min_confidence: float = 0.0,
                                      load_chunk_text: bool = True, only_validated: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Load extracted data for a specific extraction job.
        """
        extracted_data = []

        # Try filesystem storage first if enabled
        if self.enable_filesystem_storage and self.filesystem_storage_manager:
            try:
                logger.info(f"Attempting to load extracted data from filesystem for job {extraction_job_id}")

                # Load results from the specific job's files
                results, _, _ = self.filesystem_storage_manager.retrieve_results(
                    job_id=extraction_job_id,
                    page=1,
                    page_size=10000,  # Large page size to get all results
                    min_confidence=min_confidence
                )

                for result in results:
                    # Apply validation filter if requested
                    if only_validated and not result.get('metadata', {}).get('is_validated', False):
                        continue

                    # Convert filesystem format to standardized format
                    chunk_id = result.get('chunk_id', '')
                    chunk_text = result.get('chunk_text', '')
                    # If chunk_text is missing and we need it, load from chunk file
                    if not chunk_text and chunk_id and load_chunk_text:
                        chunk_text = self.load_chunk_text(project_id, chunk_id)

                    # Extract entities from metadata.extracted_entities
                    entities = []
                    metadata = result.get('metadata', {})
                    if 'extracted_entities' in metadata:
                        for category_data in metadata['extracted_entities'].values():
                            if 'entities' in category_data:
                                entities.extend(category_data['entities'])
                            if 'texts' in category_data:
                                entities.extend(category_data['texts'])

                    item = {
                        'chunk_id': chunk_id,
                        'categories': result.get('categories_matched', []),
                        'confidence': result.get('confidence_score', 0.0),
                        'chunk_text': chunk_text,
                        'classifications': result.get('classifications', {}),
                        'metadata': metadata,
                        'entities': entities,  # Add extracted entities
                        'source_type': 'extraction',
                        'extraction_job_id': extraction_job_id
                    }
                    extracted_data.append(item)

                if extracted_data:
                    logger.info(f"Loaded {len(extracted_data)} extraction results from filesystem for job {extraction_job_id}")
                    return extracted_data

            except Exception as e:
                logger.warning(f"Filesystem loading failed for job {extraction_job_id}: {e}")

        # Fallback to database if filesystem failed or disabled
        if self.extraction_result_repository:
            try:
                logger.info(f"Falling back to database loading for job {extraction_job_id}")

                # Get all extraction results for the specific job
                results, _ = self.extraction_result_repository.get_results_by_job(
                    job_id=extraction_job_id,
                    page=1,
                    page_size=10000,  # Large page size to get all results
                    min_confidence=min_confidence
                )

                if not results:
                    logger.info(f"No extraction results found for job {extraction_job_id}")
                    return None

                # Convert database results to usable format
                for result in results:
                    # Parse JSON fields - result is a dict from database query
                    try:
                        categories = json.loads(result['categories']) if result.get('categories') else []
                        confidence = float(result['confidence']) if result.get('confidence') is not None else 0.0
                        extracted_data_dict = json.loads(result['extracted_data']) if result.get('extracted_data') else {}
                        
                        # Apply validation filter if requested
                        if only_validated and not extracted_data_dict.get('is_validated', False):
                            continue

                        # Create a standardized format for dataset generation
                        chunk_id = str(result['chunk_id']) if result.get('chunk_id') else ''
                        chunk_text = extracted_data_dict.get('chunk_text', '')
                        # If chunk_text is missing and we need it, load from chunk file
                        if not chunk_text and chunk_id and load_chunk_text:
                            chunk_text = self.load_chunk_text(project_id, chunk_id)

                        # Extract entities from metadata.extracted_entities
                        entities = []
                        if 'extracted_entities' in extracted_data_dict.get('metadata', {}):
                            for category_data in extracted_data_dict['metadata']['extracted_entities'].values():
                                if 'entities' in category_data:
                                    entities.extend(category_data['entities'])
                                if 'texts' in category_data:
                                    entities.extend(category_data['texts'])

                        item = {
                            'chunk_id': chunk_id,
                            'categories': categories,
                            'confidence': confidence,
                            'chunk_text': chunk_text,
                            'classifications': extracted_data_dict.get('classifications', {}),
                            'metadata': extracted_data_dict.get('metadata', {}),
                            'entities': entities,  # Add extracted entities
                            'source_type': 'extraction',
                            'extraction_job_id': extraction_job_id
                        }
                        extracted_data.append(item)
                    except (json.JSONDecodeError, IndexError, TypeError) as e:
                        logger.warning(f"Failed to parse extraction result {result}: {e}")
                        continue

                logger.info(f"Loaded {len(extracted_data)} extraction results from database for job {extraction_job_id}")
                return extracted_data

            except Exception as e:
                logger.error(f"Failed to load extracted data from database for job {extraction_job_id}: {e}")
                return None

        logger.warning("No extraction result repository available, cannot load extracted data")
        return None

    def get_category_names_from_db(self, category_ids: List[str]) -> List[str]:
        """Get category names from database using category IDs."""
        if not self.taxonomy_loader:
            return category_ids  # Fallback to using IDs as names

        try:
            category_names = []
            for cat_id in category_ids:
                # Query taxonomy to get category name by ID
                category_info = self.taxonomy_loader.get_category_by_id(cat_id)
                if category_info and 'name' in category_info:
                    category_names.append(category_info['name'])
                else:
                    category_names.append(cat_id)  # Fallback to ID if name not found
            return category_names
        except Exception as e:
            logger.warning(f"Failed to get category names from database: {e}")
            return category_ids  # Fallback to using IDs

    def load_chunk_text(self, project_id: int, chunk_id: str) -> str:
        """
        Load chunk text from the chunk file using chunk_id.
        Thread-safe implementation using locks to prevent file descriptor corruption.
        """
        # Use lock to prevent concurrent file access issues
        with self._file_lock:
            try:
                # Query database to get the correct file path for this chunk_id
                db_path = "storage/database.db"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Convert chunk_id to int if it's a string, or handle as string UUID if needed
                try:
                    chunk_id_val = int(chunk_id)
                except (ValueError, TypeError):
                    chunk_id_val = str(chunk_id)

                cursor.execute("SELECT file_path FROM chunks WHERE id = ?", (chunk_id_val,))
                row = cursor.fetchone()
                conn.close()

                if row and row[0]:
                    chunk_path = row[0]
                    if os.path.exists(chunk_path):
                        with open(chunk_path, 'r', encoding='utf-8') as f:
                            return f.read().strip()
                    else:
                        logger.warning(f"Chunk file not found: {chunk_path}")
                        return ""
                else:
                    logger.warning(f"No file_path found for chunk_id {chunk_id}")
                    return ""

            except Exception as e:
                logger.warning(f"Failed to load chunk text for {chunk_id}: {e}")
                return ""

    def load_raw_extraction_file(self, project_id: int, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Load raw extraction file data.
        """
        try:
            extraction_file_path = os.path.join("storage", "extract", str(project_id), f"extraction_{job_id}_*.json")
            matching_files = glob.glob(extraction_file_path)
            if matching_files:
                with open(matching_files[0], 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load raw extraction file for job {job_id}: {e}")
        return None
