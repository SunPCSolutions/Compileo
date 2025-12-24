import concurrent.futures
from typing import Union, List, Dict, Any, Optional, Tuple, Callable
import threading

from .prompt_builder import PromptBuilder
from .output_formatter import OutputFormatter
from .llm_interaction import LLMInteraction
from .evaluation_dataset_generator import EvaluationDatasetGenerator
from .version_manager import DatasetVersionManager
from src.compileo.storage.src.project.database_repositories import (
    DocumentRepository, ExtractionResultRepository, ExtractionJobRepository
)
from src.compileo.features.taxonomy.loader import TaxonomyLoader
from src.compileo.features.extraction.filesystem_storage import (
    ExtractionFileManager, FilesystemStorageManager, HybridStorageManager
)
from typing import Optional
import json
import signal
import time
from ...core.logging import get_logger

logger = get_logger(__name__)
from contextlib import contextmanager
import os


class DatasetGenerator:
    """
    Generates a dataset using a prompt builder and formats the output.
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        output_formatter: OutputFormatter,
        llm_interaction: LLMInteraction,
        document_repository: DocumentRepository,
        extraction_result_repository: Optional[ExtractionResultRepository] = None,
        taxonomy_loader: Optional[TaxonomyLoader] = None,
        version_manager: Optional[DatasetVersionManager] = None,
        filesystem_storage_manager: Optional[FilesystemStorageManager] = None,
        hybrid_storage_manager: Optional[HybridStorageManager] = None,
        enable_filesystem_storage: bool = True,
        extraction_job_repository: Optional[ExtractionJobRepository] = None,
    ):
        """
        Initializes the DatasetGenerator.

        Args:
            prompt_builder: An instance of PromptBuilder.
            output_formatter: An instance of OutputFormatter.
            llm_interaction: An instance of LLMInteraction.
            document_repository: An instance of DocumentRepository.
            extraction_result_repository: An instance of ExtractionResultRepository for extraction-based generation.
            taxonomy_loader: An instance of TaxonomyLoader for taxonomy-driven generation.
            version_manager: An instance of DatasetVersionManager for dataset versioning.
            filesystem_storage_manager: Filesystem storage manager for extraction results.
            hybrid_storage_manager: Hybrid storage manager combining filesystem and database.
            enable_filesystem_storage: Whether to enable filesystem storage for extraction results.
            extraction_job_repository: Repository for accessing extraction jobs.
        """
        self.prompt_builder = prompt_builder
        self.output_formatter = output_formatter
        self.llm_interaction = llm_interaction
        self.document_repository = document_repository
        self.extraction_result_repository = extraction_result_repository
        self.taxonomy_loader = taxonomy_loader
        self.version_manager = version_manager
        self.filesystem_storage_manager = filesystem_storage_manager
        self.hybrid_storage_manager = hybrid_storage_manager
        self.enable_filesystem_storage = enable_filesystem_storage
        self.extraction_job_repository = extraction_job_repository

        # Thread lock for file operations to prevent concurrent file descriptor issues
        self._file_lock = threading.Lock()

        # Initialize filesystem storage if not provided but enabled
        if self.enable_filesystem_storage and not self.filesystem_storage_manager:
            file_manager = ExtractionFileManager()
            self.filesystem_storage_manager = FilesystemStorageManager(file_manager)

        # Initialize hybrid storage if not provided
        if not self.hybrid_storage_manager and self.filesystem_storage_manager and self.extraction_result_repository:
            self.hybrid_storage_manager = HybridStorageManager(
                self.filesystem_storage_manager,
                self.extraction_result_repository,
                enable_filesystem=self.enable_filesystem_storage
            )

    def load_extracted_data(self, project_id: int, min_confidence: float = 0.0,
                            use_filesystem: bool = True, load_chunk_text: bool = True,
                            only_validated: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Load extracted data for a project from filesystem (preferred) or database (fallback).

        Args:
            project_id: The project ID to load extraction results for.
            min_confidence: Minimum confidence threshold for results.
            use_filesystem: Whether to prefer filesystem storage over database.
            load_chunk_text: Whether to load original chunk text.
            only_validated: Whether to only include results that have passed validation.

        Returns:
            List of extracted data items, or None if no extraction results exist.
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
                                    chunk_text = self._load_chunk_text(project_id, chunk_id)

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
                            chunk_text = self._load_chunk_text(project_id, chunk_id)

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

    def _get_category_names_from_db(self, category_ids: List[str]) -> List[str]:
        """Get category names from database using category IDs."""
        if not hasattr(self, 'taxonomy_loader') or not self.taxonomy_loader:
            return category_ids  # Fallback to using IDs as names

        try:
            category_names = []
            for cat_id in category_ids:
                # Query taxonomy to get category name by ID
                # This assumes taxonomy_loader has a method to get category by ID
                category_info = self.taxonomy_loader.get_category_by_id(cat_id)
                if category_info and 'name' in category_info:
                    category_names.append(category_info['name'])
                else:
                    category_names.append(cat_id)  # Fallback to ID if name not found
            return category_names
        except Exception as e:
            logger.warning(f"Failed to get category names from database: {e}")
            return category_ids  # Fallback to using IDs

    def _load_chunk_text(self, project_id: int, chunk_id: str) -> str:
        """
        Load chunk text from the chunk file using chunk_id.
        Thread-safe implementation using locks to prevent file descriptor corruption.

        Args:
            project_id: The project ID
            chunk_id: The chunk ID (database ID, not filename)

        Returns:
            The chunk text content, or empty string if not found
        """
        # Use lock to prevent concurrent file access issues
        with self._file_lock:
            try:
                # Query database to get the correct file path for this chunk_id
                import sqlite3
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

    def load_extracted_data_batch(self, project_id: int, min_confidence: float = 0.0,
                                  category_filter: Optional[str] = None,
                                  limit: Optional[int] = None, offset: Optional[int] = 0,
                                  use_filesystem: bool = True, load_chunk_text: bool = True,
                                  only_validated: bool = False,
                                  progress_callback: Optional[callable] = None) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Load extracted data in batches with pagination and filtering.

        Args:
            project_id: The project ID to load extraction results for.
            min_confidence: Minimum confidence threshold for results.
            category_filter: Filter by category name.
            limit: Maximum number of results to return.
            offset: Number of results to skip.
            use_filesystem: Whether to prefer filesystem storage.
            load_chunk_text: Whether to load original chunk text.
            only_validated: Whether to only include results that have passed validation.
            progress_callback: Optional callback for progress updates.

        Returns:
            Tuple of (results, has_more) where has_more indicates if there are more results available.
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
                                    chunk_text = self._load_chunk_text(project_id, chunk_id)

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
                            chunk_text = self._load_chunk_text(project_id, chunk_id)

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

        Args:
            project_id: The project ID to load extraction results for.
            extraction_job_id: The specific extraction job ID to load results for.
            min_confidence: Minimum confidence threshold for results.
            load_chunk_text: Whether to load chunk text for each result.
            only_validated: Whether to only include results that have passed validation.

        Returns:
            List of extracted data items for the specific job, or None if no results exist.
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
                        chunk_text = self._load_chunk_text(project_id, chunk_id)

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
                            chunk_text = self._load_chunk_text(project_id, chunk_id)

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

    def generate_from_extracted(self, project_id: int, extracted_data: List[Dict[str, Any]], prompt_name: str,
                               format_type: str, concurrency: int = 1,
                               taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                               datasets_per_chunk: int = 3) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generate dataset from extracted data with categorized content.

        Args:
            extracted_data: List of extracted data items with categories and content.
            prompt_name: The name of the prompt to use.
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            taxonomy_project: Optional taxonomy project name.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Maximum number of datasets to generate per extracted item.

        Returns:
            The generated and formatted dataset.
        """
        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            # Use concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                from functools import partial
                generate_func = partial(self._generate_qa_for_extracted_item, project_id, prompt_name,
                                      taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                      datasets_per_chunk=datasets_per_chunk)
                results = list(executor.map(generate_func, extracted_data))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            # Sequential processing
            for item in extracted_data:
                try:
                    qa_pairs = self._generate_qa_for_extracted_item(project_id, prompt_name, item,
                                                                   taxonomy_project=taxonomy_project,
                                                                   taxonomy_name=taxonomy_name,
                                                                   datasets_per_chunk=datasets_per_chunk)
                    llm_generated_content.extend(qa_pairs)
                except Exception as exc:
                    logger.warning(f"Extracted item {item.get('chunk_id', 'unknown')} generated an exception: {exc}")

        formatted_dataset = self.output_formatter.format_dataset(
            llm_generated_content, format_type
        )

        # Trigger post-generation hooks (benchmarking, etc.)
        self._trigger_post_generation_hooks(project_id, llm_generated_content, {
            'prompt_name': 'qa_pairs',
            'format_type': format_type,
            'concurrency': concurrency,
            'taxonomy_project': taxonomy_project,
            'taxonomy_name': taxonomy_name,
            'generation_method': 'qa_pairs'
        })

        return formatted_dataset


    def load_raw_extraction_file(self, project_id: int, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Load raw extraction file data.

        Args:
            project_id: The project ID
            job_id: The extraction job ID

        Returns:
            Raw extraction file data or None if not found
        """
        try:
            import os
            extraction_file_path = os.path.join("storage", "extract", str(project_id), f"extraction_{job_id}_*.json")
            import glob
            matching_files = glob.glob(extraction_file_path)
            if matching_files:
                with open(matching_files[0], 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load raw extraction file for job {job_id}: {e}")
        return None

    def generate_from_extracted_entities_only(self, project_id: int, prompt_name: str,
                                            format_type: str, concurrency: int = 1,
                                            datasets_per_chunk: int = 3, extraction_file_id: Optional[str] = None,
                                            custom_prompt: Optional[str] = None, generation_mode: str = "question and answer",
                                            selected_categories: Optional[List[str]] = None,
                                            only_validated: bool = False,
                                            progress_callback: Optional[Callable[[float, str], None]] = None) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generate dataset from extracted entities text content (not using chunk text).

        This mode generates datasets by processing the actual extracted text content from
        extracted_entities.texts arrays, similar to how Taxonomy mode processes chunks.
        It creates questions and answers based on the extracted content.

        Args:
            project_id: The project ID.
            prompt_name: The name of the prompt to use.
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            datasets_per_chunk: Maximum number of datasets to generate per extracted item.
            extraction_file_id: Optional specific extraction job ID to filter by.
            custom_prompt: Optional custom prompt to use instead of prompt_name.
            only_validated: Whether to only include results that have passed validation.

        Returns:
            The generated and formatted dataset.
        """
        # Load extracted data for the project
        extracted_data = []

        # Extract mode NEVER loads chunk text - it works purely with extracted entities
        load_chunk_text = False

        # If extraction_file_id is provided, load from raw extraction file
        if extraction_file_id is not None:
            raw_extraction_data = self.load_raw_extraction_file(project_id, extraction_file_id)
            if raw_extraction_data:
                # Convert raw extraction file format to standardized format
                for result in raw_extraction_data.get('results', []):
                    # In new schema, metadata IS extracted_data from TaxonomyExtractor
                    metadata = result.get('metadata', {})
                    
                    # Apply validation filter if requested
                    if only_validated and not metadata.get('is_validated', False):
                        continue

                    entities = []
                    if 'extracted_entities' in metadata:
                        for category_data in metadata['extracted_entities'].values():
                            if 'entities' in category_data:
                                entities.extend(category_data['entities'])
                            if 'texts' in category_data:
                                entities.extend(category_data['texts'])

                    item = {
                        'chunk_id': result.get('chunk_id', ''),
                        'categories': result.get('categories_matched', []),
                        'confidence': result.get('confidence_score', 0.0),
                        'chunk_text': '',  # Extract mode never uses chunk text
                        'classifications': result.get('classifications', {}),
                        'metadata': metadata,
                        'entities': entities,
                        'source_type': 'extraction',
                        'extraction_job_id': extraction_file_id
                    }
                    extracted_data.append(item)

                if not extracted_data:
                    raise ValueError(f"No extracted data found in extraction file {extraction_file_id}")
            else:
                raise ValueError(f"Could not load extraction file {extraction_file_id} for project {project_id}")
        else:
            # Load extracted data for the project (Extract mode never loads chunk text)
            extracted_data = self.load_extracted_data(project_id, load_chunk_text=False, only_validated=only_validated)
            if not extracted_data:
                raise ValueError(f"No extracted data found for project {project_id}")

        # Get category names from database if selected_categories provided
        category_names = []
        if selected_categories:
            category_names = self._get_category_names_from_db(selected_categories)

        # Aggregate texts by category across all extracted items
        # This ensures datasets_per_chunk acts as "datasets per category"
        category_map = {}  # category_name -> {'texts': [], 'entities': []}

        logger.info(f"Processing {len(extracted_data)} extracted items for category aggregation")

        for item in extracted_data:
            metadata = item.get('metadata', {})

            if 'extracted_entities' in metadata:
                for category_id, category_data in metadata['extracted_entities'].items():
                    category_name = category_data.get('category_name', category_id)

                    # Filter by selected categories if provided
                    if category_names and category_name not in category_names:
                        continue

                    texts = category_data.get('texts', [])
                    entities = category_data.get('entities', [])

                    # Initialize category in map if not present
                    if category_name not in category_map:
                        category_map[category_name] = {'texts': [], 'entities': []}
                    
                    # Aggregate texts if present
                    if texts:
                        category_map[category_name]['texts'].extend(texts)
                    
                    # Aggregate entities if present
                    if entities:
                        category_map[category_name]['entities'].extend(entities)

        # Create synthetic items for each category to process
        # This changes the unit of work from "extraction result" to "category"
        items_to_process = []
        for cat_name, data in category_map.items():
            texts = data['texts']
            entities = data['entities']
            
            # Skip if no content found for this category
            if not texts and not entities:
                continue
                
            # Create a synthetic item that focuses on this single category
            synthetic_item = {
                'chunk_id': f"category_{cat_name}",
                'categories': [cat_name],
                'confidence': 1.0,
                'metadata': {
                    'extraction_type': 'aggregated',
                    'extracted_entities': {
                        cat_name: {
                            'category_name': cat_name,
                            'texts': texts,
                            'entities': entities
                        }
                    }
                },
                'entities': entities  # Also set at top level for compatibility
            }
            items_to_process.append(synthetic_item)
            
        logger.info(f"Created {len(items_to_process)} synthetic items for processing from {len(category_map)} categories")

        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            # Use concurrent processing over CATEGORIES
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                from functools import partial
                # We don't need to pass category_names filter again as we already filtered during aggregation
                generate_func = partial(self._generate_content_from_extracted_data, project_id, prompt_name,
                                      datasets_per_chunk=datasets_per_chunk, custom_prompt=custom_prompt,
                                      generation_mode=generation_mode, category_names=None)
                results = list(executor.map(generate_func, items_to_process))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            # Sequential processing over CATEGORIES
            total_items = len(items_to_process)
            for i, item in enumerate(items_to_process):
                if progress_callback:
                    # Update progress within the 50-95% range
                    p = 0.5 + (i / total_items) * 0.45
                    cat_name = item.get('categories', ['unknown'])[0]
                    progress_callback(p, f"Processing category {i+1}/{total_items}: {cat_name}...")

                try:
                    qa_pairs = self._generate_content_from_extracted_data(project_id, prompt_name, item,
                                                                          datasets_per_chunk=datasets_per_chunk,
                                                                          custom_prompt=custom_prompt,
                                                                          generation_mode=generation_mode,
                                                                          category_names=None)
                    llm_generated_content.extend(qa_pairs)
                except Exception as exc:
                    logger.warning(f"Category item {item.get('chunk_id', 'unknown')} generated an exception: {exc}")

        formatted_dataset = self.output_formatter.format_dataset(
            llm_generated_content, format_type
        )

        # Trigger post-generation hooks (benchmarking, etc.)
        self._trigger_post_generation_hooks(project_id, llm_generated_content, {
            'prompt_name': prompt_name,
            'format_type': format_type,
            'concurrency': concurrency,
            'taxonomy_project': None,
            'taxonomy_name': None,
            'generation_method': 'qa_pairs'
        })

        return formatted_dataset

    def _generate_qa_for_extracted_item(self, project_id: int, prompt_name: str, extracted_item: Dict[str, Any],
                                       taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                       datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generate Q&A pairs for a single extracted item using its categorized content.

        Args:
            prompt_name: The prompt name to use.
            extracted_item: The extracted data item with categories and content.
            taxonomy_project: Optional taxonomy project.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Number of datasets to generate per item.

        Returns:
            List of generated Q&A pairs.
        """
        results = []
        chunk_text = extracted_item.get('chunk_text', '')
        categories = extracted_item.get('categories', [])

        # Create enhanced prompt using categories for better context
        category_context = f"Content categories: {', '.join(categories)}" if categories else ""

        for i in range(datasets_per_chunk):
            # Build prompt with category context
            final_prompt = self.prompt_builder.build_prompt(
                project_id=project_id,  # Use the actual project_id
                prompt_name=prompt_name,
                chunk_text=chunk_text,
                taxonomy_project=taxonomy_project,
                taxonomy_name=taxonomy_name,
                datasets_per_chunk=datasets_per_chunk
            )

            # Add category context to the prompt
            if category_context:
                final_prompt = f"{category_context}\n\n{final_prompt}"

            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs from this categorized content. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    item["categories"] = categories
                    item["confidence"] = extracted_item.get('confidence', 0.0)
                    item["source_type"] = "extraction"
                    results.append(item)
            else:
                result["categories"] = categories
                result["confidence"] = extracted_item.get('confidence', 0.0)
                result["source_type"] = "extraction"
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def _generate_content_from_extracted_data(self, project_id: int, prompt_name: str, extracted_item: Dict[str, Any],
                                                 datasets_per_chunk: int = 3, custom_prompt: Optional[str] = None,
                                                 generation_mode: str = "question and answer", category_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
            """
            Generate content from extracted data, handling both NER and full text extraction types.
    
            This method intelligently processes extracted data based on what's available:
            - For full text extractions: Uses extracted text content from texts arrays
            - For NER extractions: Falls back to entity-based generation
            - Routes based on generation mode (question/answer, summarization, instruction following, etc.)
    
            Args:
                project_id: The project ID.
                prompt_name: The prompt name to use.
                extracted_item: The extracted data item with entities and/or text content.
                datasets_per_chunk: Number of datasets to generate per item.
                custom_prompt: Optional custom prompt to use.
                generation_mode: The generation mode (question and answer, summarization, etc.).
    
            Returns:
                List of generated content items.
            """
            results = []
            categories = extracted_item.get('categories', [])
            metadata = extracted_item.get('metadata', {})
    
            # Determine extraction type and available content
            extraction_type = metadata.get('extraction_type', 'unknown')
    
            # Build category context from provided category_names if available
            category_context = ""
            if category_names:
                category_context = f"Focus on these categories: {', '.join(category_names)}. "
    
            if 'extracted_entities' in metadata:
                for category_id, category_data in metadata['extracted_entities'].items():
                    category_name = category_data.get('category_name', category_id)
                    if category_names and category_name not in category_names:
                        continue
                    texts = category_data.get('texts', [])
                    entities = category_data.get('entities', [])
    
                    if not texts and not entities:
                        logger.warning(f"Category {category_name} has no texts or entities, skipping")
                        continue
    
                    # Combine texts for this category
                    combined_text = "\n\n".join(texts)
    
                    # Generate for this category's content
                    for i in range(datasets_per_chunk):
                        # Build prompt with category-specific context
                        category_specific_context = f"Category: {category_name}. {category_context}"
    
                        if combined_text.strip():
                            final_prompt = self._build_prompt_from_text_content(
                                combined_text, generation_mode, custom_prompt, prompt_name, project_id, datasets_per_chunk, category_specific_context
                            )
                        else:
                            # Fallback to entity-based if no text
                            final_prompt = self._build_prompt_from_entities(
                                extracted_item, generation_mode, custom_prompt, prompt_name, project_id, datasets_per_chunk, category_specific_context
                            )
    
                        result = self.llm_interaction.generate(final_prompt)
    
                        # Handle extract mode: if result is dict with answer as JSON string, expand it
                        expanded_items = []
                        if isinstance(result, dict) and isinstance(result.get('answer'), str):
                            try:
                                parsed = json.loads(result['answer'])
                                if isinstance(parsed, list):
                                    for item in parsed[:datasets_per_chunk]:
                                        if item and isinstance(item, dict):
                                            item_copy = item.copy()
                                            item_copy["categories"] = [category_name]
                                            item_copy["confidence"] = extracted_item.get('confidence', 0.0)
                                            item_copy["extraction_type"] = extraction_type
                                            item_copy["source_type"] = "extract_adaptive"
                                            expanded_items.append(item_copy)
                            except (json.JSONDecodeError, KeyError):
                                pass
    
                        if expanded_items:
                            results.extend(expanded_items)
                        else:
                            # Normal handling
                            if isinstance(result, list):
                                for item in result[:datasets_per_chunk]:
                                    if item and isinstance(item, dict):
                                        item_copy = item.copy()
                                        item_copy["categories"] = [category_name]
                                        item_copy["confidence"] = extracted_item.get('confidence', 0.0)
                                        item_copy["extraction_type"] = extraction_type
                                        item_copy["source_type"] = "extract_adaptive"
                                        
                                        # Map instruction following fields for fine-tuning appropriateness
                                        if generation_mode == "instruction following":
                                            item_copy.setdefault("instruction", item_copy.get("instruction", ""))
                                            item_copy.setdefault("input", item_copy.get("input", ""))
                                            item_copy.setdefault("output", item_copy.get("output", ""))
                                            
                                        results.append(item_copy)
                            elif isinstance(result, dict) and result:
                                item_copy = result.copy()
                                item_copy["categories"] = [category_name]
                                item_copy["confidence"] = extracted_item.get('confidence', 0.0)
                                item_copy["extraction_type"] = extraction_type
                                item_copy["source_type"] = "extract_adaptive"
                                
                                # Map specialized fields based on mode and clean up unrelated fields
                                final_item = {
                                    "categories": [category_name],
                                    "confidence": extracted_item.get('confidence', 0.0),
                                    "extraction_type": extraction_type,
                                    "source_type": "extract_adaptive",
                                    "reasoning": item_copy.get("reasoning", ""),
                                    "confidence_score": item_copy.get("confidence_score", 0.0),
                                    "reasoning_steps": item_copy.get("reasoning_steps", []),
                                    "generation_mode": generation_mode
                                }

                                if generation_mode == "instruction following":
                                    final_item.update({
                                        "instruction": item_copy.get("instruction", ""),
                                        "input": item_copy.get("input", ""),
                                        "output": item_copy.get("output", "")
                                    })
                                elif generation_mode == "question":
                                    # Support both singular and plural from LLM
                                    val = item_copy.get("question") or item_copy.get("questions")
                                    if isinstance(val, list):
                                        final_item["questions"] = val
                                    elif val:
                                        final_item["question"] = val
                                    else:
                                        final_item["question"] = ""
                                elif generation_mode == "answer":
                                    # Support both singular and plural from LLM
                                    val = item_copy.get("answer") or item_copy.get("answers")
                                    if isinstance(val, list):
                                        final_item["answers"] = val
                                    elif val:
                                        final_item["answer"] = val
                                    else:
                                        final_item["answer"] = ""
                                elif generation_mode == "summarization":
                                    final_item.update({
                                        "summary": item_copy.get("summary", ""),
                                        "key_points": item_copy.get("key_points", [])
                                    })
                                else:
                                    # Default Q&A mode
                                    final_item.update({
                                        "question": item_copy.get("question", ""),
                                        "answer": item_copy.get("answer", "")
                                    })
                                    
                                results.append(final_item)
    
                            if len(results) >= datasets_per_chunk:
                                break
            
            # Default return at the end of the method
            return results[:datasets_per_chunk]
    
    def _build_prompt_from_text_content(self, text_content: str, generation_mode: str,
                                           custom_prompt: Optional[str], prompt_name: str,
                                           project_id: Union[int, str], datasets_per_chunk: int, category_context: str = "") -> str:
            """Build prompt from extracted text content."""
            base_prompt = ""
            if custom_prompt and custom_prompt.strip():
                if "{chunk}" in custom_prompt:
                    base_prompt = custom_prompt.replace("{chunk}", text_content)
                else:
                    base_prompt = f"Content: {text_content}\n\n" + custom_prompt
                # Add instruction to base on content
                base_prompt += "\n\nBase the generated content strictly on the provided text above. Use any examples only as format guidelines. Do not copy the examples; create new content derived from the provided text."
            else:
                # Convert project_id to int for prompt_builder if it's a numeric string
                proj_id_int = int(project_id) if isinstance(project_id, str) and project_id.isdigit() else project_id
                
                if generation_mode == "question":
                    base_prompt = self.prompt_builder.build_question_generation_prompt(
                        proj_id_int, text_content, None, None, datasets_per_chunk # type: ignore
                    )
                elif generation_mode == "answer":
                    base_prompt = self.prompt_builder.build_answer_generation_prompt(
                        proj_id_int, text_content, None, None, datasets_per_chunk # type: ignore
                    )
                elif generation_mode == "summarization":
                    base_prompt = self.prompt_builder.build_summarization_prompt(
                        proj_id_int, text_content, None, None, datasets_per_chunk # type: ignore
                    )
                elif generation_mode == "instruction following":
                    base_prompt = self._build_instruction_following_prompt(
                        text_content, "explain", None, None
                    )
                else:  # default or "question and answer"
                    base_prompt = self.prompt_builder.build_prompt(
                        project_id=proj_id_int, # type: ignore
                        prompt_name=prompt_name,
                        chunk_text=text_content,
                        taxonomy_project=None,
                        taxonomy_name=None,
                        datasets_per_chunk=datasets_per_chunk
                    )
    
            # Add category context if provided
            if category_context:
                base_prompt = f"{category_context}\n\n{base_prompt}"
    
            logger.debug(f"base_prompt for mode {generation_mode} length: {len(base_prompt)}")

            # Add JSON instructions
            if generation_mode == "instruction following":
                json_instructions = """\n\n
            IMPORTANT: You must respond with a valid machine-readable structure with exactly this schema:
            {
              "instruction": "A clear, actionable instruction related to the content",
              "input": "",
              "output": "A comprehensive, helpful response to the instruction",
              "reasoning": "Brief explanation of your reasoning",
              "confidence_score": 0.8,
              "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
            }

            CRITICAL CONSTRAINTS:
            - ONLY use information and concepts that are explicitly present in the provided content.
            - DO NOT draw from external knowledge, general knowledge, or information not contained in the content.
            - DO NOT generate synthetic or hypothetical content.
            - If the content does not contain sufficient information for an instruction-response pair, return an empty array [].
            - Base all instructions and responses strictly on the content provided.

            Return ONLY the raw object or array, no additional text or explanation.
            """
            elif generation_mode in ["question", "answer", "summarization"]:
                # These modes already have specialized instructions in their templates
                json_instructions = ""
            else:
                json_instructions = """\n\n
            IMPORTANT: You must respond with a valid machine-readable structure with exactly this schema:
            {
              "question": "The question you generated based on the content",
              "answer": "The answer to the question",
              "reasoning": "Brief explanation of your reasoning",
              "confidence_score": 0.8,
              "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
            }
    
            CRITICAL CONSTRAINTS:
            - ONLY use information explicitly present in the provided content.
            - Generate questions that are directly answerable from the content provided.
            - Base answers strictly on the content provided.
            - If the content doesn't support generating meaningful questions and answers, return an empty array [].
    
            Return ONLY the raw object or array, no additional text or explanation.
            """
            
            final_prompt = base_prompt + json_instructions
            
            if datasets_per_chunk > 1 and generation_mode not in ["question", "answer", "summarization"]:
                if generation_mode == "instruction following":
                    final_prompt += f"\n\nGenerate {datasets_per_chunk} different instruction-response pairs based on the content provided above. Return them as a machine-readable array of objects."
                else:
                    final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs based on the content provided above. Return them as a machine-readable array of objects."
            
            logger.debug(f"Final prompt for dataset generation: {final_prompt}")
            return final_prompt
    
    def _build_prompt_from_entities(self, extracted_item: Dict[str, Any], generation_mode: str,
                                   custom_prompt: Optional[str], prompt_name: str,
                                   project_id: int, datasets_per_chunk: int, category_context: str = "") -> str:
        """Build prompt from extracted entities (for NER extractions)."""
        entities = extracted_item.get('entities', [])
        entity_text = ", ".join(entities) if entities else "various concepts"

        if generation_mode == "question":
            return self.prompt_builder.build_question_generation_prompt(
                project_id, f"Focus on these concepts: {entity_text}", None, None, datasets_per_chunk
            )
        elif generation_mode == "answer":
            return self.prompt_builder.build_answer_generation_prompt(
                project_id, f"Focus on these concepts: {entity_text}", None, None, datasets_per_chunk
            )
        elif generation_mode == "summarization":
            return self.prompt_builder.build_summarization_prompt(
                project_id, f"Focus on these concepts: {entity_text}", None, None, datasets_per_chunk
            )
        elif generation_mode == "instruction following":
            return self._build_instruction_following_prompt(
                f"Focus on these concepts: {entity_text}", "explain", None, None
            )
        else:  # default or "question and answer"
            if custom_prompt and custom_prompt.strip():
                if "{chunk}" in custom_prompt:
                    final_prompt = custom_prompt.replace("{chunk}", f"Focus on these concepts: {entity_text}")
                else:
                    final_prompt = f"Content: Focus on these concepts: {entity_text}\n\n" + custom_prompt
                # Add category context if provided
                if category_context:
                    final_prompt = f"{category_context}\n\n{final_prompt}"
                if generation_mode == "instruction following":
                    json_instructions = """

IMPORTANT: You must respond with a valid JSON object (or array of objects) with exactly this structure:
{
  "instruction": "A clear, actionable instruction related to the content",
  "input": "",
  "output": "A comprehensive, helpful response to the instruction",
  "reasoning": "Brief explanation of your reasoning",
  "confidence_score": 0.8,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}

CRITICAL CONSTRAINTS:
- ONLY use information and concepts that are explicitly present in the provided content.
- DO NOT draw from external knowledge, general knowledge, or information not contained in the content.
- DO NOT generate synthetic or hypothetical content.
- If the content does not contain sufficient information for an instruction-response pair, return an empty array [].
- Base all instructions and responses strictly on the content provided.

Return ONLY the JSON object or array, no additional text or explanation."""
                else:
                    json_instructions = """

IMPORTANT: You must respond with a valid machine-readable structure with exactly this schema:
{
  "question": "The question you generated based on the content",
  "answer": "The answer to the question",
  "reasoning": "Brief explanation of your reasoning",
  "confidence_score": 0.8,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}

CRITICAL CONSTRAINTS:
- ONLY use information explicitly present in the provided content.
- Generate questions that are directly answerable from the content provided.
- Base answers strictly on the content provided.
- If the content doesn't support generating meaningful questions and answers, return an empty array [].

Return ONLY the raw object or array, no additional text or explanation."""

                final_prompt += json_instructions

                if datasets_per_chunk > 1 and generation_mode not in ["question", "answer", "summarization"]:
                    if generation_mode == "instruction following":
                        final_prompt += f"\n\nGenerate {datasets_per_chunk} different instruction-response pairs based on the content provided above. Return them as a machine-readable array of objects."
                    else:
                        final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs based on the content provided above. Return them as a machine-readable array of objects."
                return final_prompt
            else:
                final_prompt = self.prompt_builder.build_prompt(
                    project_id=project_id,
                    prompt_name=prompt_name,
                    chunk_text=f"Focus on these concepts: {entity_text}",
                    taxonomy_project=None,
                    taxonomy_name=None
                )
                # Add category context if provided
                if category_context:
                    final_prompt = f"{category_context}\n\n{final_prompt}"
                final_prompt += f"\n\nGenerate educational content about these concepts: {entity_text}. Create questions and answers that help users understand these concepts."
                if datasets_per_chunk > 1:
                    final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs about these concepts. Return them as a JSON array."
                return final_prompt

    def _generate_qa_for_extracted_entities_only(self, project_id: int, prompt_name: str, extracted_item: Dict[str, Any],
                                                  datasets_per_chunk: int = 3, custom_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate Q&A pairs for extracted entities only (not using chunk text).

        This method generates questions and answers based on the extracted entities
        themselves, creating educational content about the concepts/entities.

        Args:
            project_id: The project ID.
            prompt_name: The prompt name to use.
            extracted_item: The extracted data item with entities.
            datasets_per_chunk: Number of datasets to generate per item.

        Returns:
            List of generated Q&A pairs.
        """
        results = []
        entities = extracted_item.get('entities', [])
        categories = extracted_item.get('categories', [])

        # Create a text representation of the entities for the prompt
        entity_text = ", ".join(entities) if entities else "various concepts"

        for i in range(datasets_per_chunk):
            # Use custom prompt if provided, otherwise build from prompt_name
            if custom_prompt and custom_prompt.strip():
                final_prompt = custom_prompt.replace("{chunk}", f"Focus on these extracted concepts: {entity_text}")
                # Add JSON formatting instructions
                final_prompt += f"""

IMPORTANT: You must respond with a valid JSON object (or array of objects) with exactly this structure:
{{
  "question": "The question you generated based on the content",
  "answer": "The answer to the question",
  "reasoning": "Brief explanation of your reasoning",
  "confidence_score": 0.8,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}}

CRITICAL CONSTRAINTS:
- ONLY use information explicitly present in the provided content.
- Generate questions that are directly answerable from the content provided.
- Base answers strictly on the content provided.
- If the content doesn't support generating meaningful questions and answers, return an empty array [].

Return ONLY the JSON object or array, no additional text or explanation."""

                if datasets_per_chunk > 1:
                    final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs. Return them as a JSON array of objects."
            else:
                # Build prompt focused on the extracted entities using prompt_name
                final_prompt = self.prompt_builder.build_prompt(
                    project_id=project_id,
                    prompt_name=prompt_name,
                    chunk_text=f"Focus on these extracted concepts: {entity_text}",
                    taxonomy_project=None,
                    taxonomy_name=None,
                    datasets_per_chunk=datasets_per_chunk
                )

                # Modify prompt to focus on entities rather than chunk text
                final_prompt = final_prompt.replace("{chunk}", f"the following extracted concepts: {entity_text}")

                # Add instruction to generate content about the entities
                final_prompt += f"\n\nGenerate educational content about these extracted entities: {entity_text}. Create questions and answers that help users understand these concepts."

                if datasets_per_chunk > 1:
                    final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs about these concepts. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    item["categories"] = categories
                    item["entities"] = entities
                    item["confidence"] = extracted_item.get('confidence', 0.0)
                    item["source_type"] = "extract_entities_only"
                    results.append(item)
            else:
                result["categories"] = categories
                result["entities"] = entities
                result["confidence"] = extracted_item.get('confidence', 0.0)
                result["source_type"] = "extract_entities_only"
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def _generate_qa_for_chunk(self, project_id: int, prompt_name: str, chunk: str,
                              taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                              datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates multiple question-answer pairs for a single chunk.

        Args:
            project_id: The project ID.
            prompt_name: The prompt name to use.
            chunk: The text chunk to process.
            taxonomy_project: Optional taxonomy project.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Number of datasets to generate per chunk.

        Returns:
            List of generated Q&A pairs.
        """
        results = []
        for i in range(datasets_per_chunk):
            final_prompt = self.prompt_builder.build_prompt(project_id, prompt_name, chunk,
                                                            taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                                            datasets_per_chunk=datasets_per_chunk)
            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs from this text chunk. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    results.append(item)
            else:
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def _generate_questions_for_chunk(self, project_id: int, chunk: str,
                                    taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                    datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates multiple questions for a single chunk.

        Args:
            project_id: The project ID.
            chunk: The text chunk to process.
            taxonomy_project: Optional taxonomy project.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Number of questions to generate per chunk.

        Returns:
            List of generated questions.
        """
        results = []
        for i in range(datasets_per_chunk):
            final_prompt = self.prompt_builder.build_question_generation_prompt(project_id, chunk, taxonomy_project, taxonomy_name)
            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different questions from this text chunk. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    item["result_type"] = "questions"
                    results.append(item)
            else:
                result["result_type"] = "questions"
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def _generate_answers_for_chunk(self, project_id: int, chunk: str,
                                  taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                  datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates multiple answers for a single chunk.

        Args:
            project_id: The project ID.
            chunk: The text chunk to process.
            taxonomy_project: Optional taxonomy project.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Number of answers to generate per chunk.

        Returns:
            List of generated answers.
        """
        results = []
        for i in range(datasets_per_chunk):
            final_prompt = self.prompt_builder.build_answer_generation_prompt(project_id, chunk, taxonomy_project, taxonomy_name)
            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different answers from this text chunk. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    item["result_type"] = "answers"
                    results.append(item)
            else:
                result["result_type"] = "answers"
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def _generate_summary_for_chunk(self, project_id: int, chunk: str,
                                  taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                  datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates multiple summaries for a single chunk.

        Args:
            project_id: The project ID.
            chunk: The text chunk to process.
            taxonomy_project: Optional taxonomy project.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Number of summaries to generate per chunk.

        Returns:
            List of generated summaries.
        """
        results = []
        for i in range(datasets_per_chunk):
            final_prompt = self.prompt_builder.build_summarization_prompt(project_id, chunk, taxonomy_project, taxonomy_name)
            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different summaries from this text chunk. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    item["result_type"] = "summary"
                    results.append(item)
            else:
                result["result_type"] = "summary"
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def generate_dataset_with_custom_prompt(
        self, project_id: int, chunks: List[str], custom_prompt: str, generation_mode: str,
        format_type: str, concurrency: int = 1,
        taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
        datasets_per_chunk: int = 3
    ) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generates a dataset using a custom prompt provided directly.

        Args:
            project_id: The ID of the project to associate with the prompt parameters.
            chunks: A list of text chunks to process.
            custom_prompt: The custom prompt text to use.
            generation_mode: The generation mode (for metadata purposes).
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            taxonomy_project: Optional taxonomy project name.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Maximum number of datasets to generate per text chunk.

        Returns:
            The generated and formatted dataset.
        """
        if not chunks:
            raise ValueError("No chunks provided for dataset generation.")

        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            # Use concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Create partial function with fixed arguments
                from functools import partial
                generate_func = partial(self._generate_with_custom_prompt, custom_prompt,
                                      taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                      datasets_per_chunk=datasets_per_chunk)
                # Map over chunks
                results = list(executor.map(generate_func, chunks))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            # Sequential processing
            for chunk in chunks:
                try:
                    qa_pairs = self._generate_with_custom_prompt(custom_prompt, chunk,
                                                               taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                                               datasets_per_chunk=datasets_per_chunk)
                    llm_generated_content.extend(qa_pairs)
                except Exception as exc:
                    logger.error(f"Chunk generated an exception: {exc}")

        formatted_dataset = self.output_formatter.format_dataset(
            llm_generated_content, format_type
        )

        # Trigger post-generation hooks (benchmarking, etc.)
        self._trigger_post_generation_hooks(project_id, llm_generated_content, {
            'prompt_name': 'custom',
            'format_type': format_type,
            'concurrency': concurrency,
            'taxonomy_project': taxonomy_project,
            'taxonomy_name': taxonomy_name,
            'generation_method': 'custom_prompt'
        })

        return formatted_dataset

    @contextmanager
    def timeout_context(self, timeout_seconds: Optional[float] = None):
        """
        Context manager for timeout handling.

        Args:
            timeout_seconds: Timeout in seconds, None for no timeout

        Raises:
            TimeoutError: If timeout is exceeded
        """
        if timeout_seconds is None:
            yield
            return

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def generate_dataset_resumable(self, project_id: int, chunks: Optional[List[str]] = None,
                                  prompt_name: str = "qa_pairs", format_type: str = "json",
                                  concurrency: int = 1, taxonomy_project: Optional[str] = None,
                                  taxonomy_name: Optional[str] = None, dataset_name: Optional[str] = None,
                                  enable_versioning: bool = False, datasets_per_chunk: int = 3,
                                  prefer_extracted: bool = True, only_validated: bool = False,
                                  options: Optional[Dict[str, Any]] = None,
                                  progress_callback: Optional[Callable[[float, str], None]] = None,
                                  resume_state: Optional[Dict[str, Any]] = None,
                                  timeout_seconds: Optional[float] = None) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generate dataset with resumable processing, progress tracking, and timeout handling.

        Args:
            project_id: The ID of the project.
            chunks: Optional list of text chunks.
            prompt_name: The name of the prompt to use.
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            taxonomy_project: Optional taxonomy project name.
            taxonomy_name: Optional taxonomy name.
            dataset_name: Optional name for the dataset (for versioning).
            enable_versioning: Whether to create a version record.
            datasets_per_chunk: Maximum number of datasets to generate per text chunk.
            prefer_extracted: Whether to prefer extracted data over raw chunks.
            options: Optional Ollama API options.
            progress_callback: Callback for progress updates (progress: float, message: str).
            resume_state: State to resume from previous interrupted run.
            timeout_seconds: Timeout for the entire operation.

        Returns:
            The generated and formatted dataset.

        Raises:
            TimeoutError: If operation times out.
        """
        start_time = time.time()

        with self.timeout_context(timeout_seconds):
            # Initialize progress
            if progress_callback:
                progress_callback(0.0, "Initializing dataset generation...")

            # Resume from previous state if provided
            processed_chunks = resume_state.get('processed_chunks', 0) if resume_state else 0
            llm_generated_content = resume_state.get('generated_content', []) if resume_state else []

            # Try to use extracted data first if preferred
            if prefer_extracted and processed_chunks == 0:
                if progress_callback:
                    progress_callback(0.1, "Loading extracted data...")

                extracted_data = self.load_extracted_data(project_id, use_filesystem=self.enable_filesystem_storage,
                                                         load_chunk_text=False, only_validated=only_validated)
                if extracted_data:
                    if progress_callback:
                        progress_callback(0.2, f"Processing {len(extracted_data)} extracted items...")

                    # Generate from extracted data with progress tracking
                    result = self._generate_from_extracted_with_progress(
                        project_id, extracted_data, prompt_name, format_type, concurrency,
                        taxonomy_project, taxonomy_name, datasets_per_chunk,
                        progress_callback, start_time, timeout_seconds
                    )

                    if progress_callback:
                        progress_callback(1.0, "Dataset generation completed")

                    return result

            # Fallback to chunk-based generation
            if not chunks:
                raise ValueError("No chunks provided for dataset generation and no extracted data available.")

            if progress_callback:
                progress_callback(0.1, f"Processing {len(chunks)} text chunks...")

            # Resume processing from where we left off
            remaining_chunks = chunks[processed_chunks:]
            total_chunks = len(chunks)

            if concurrency > 1:
                # Concurrent processing with progress tracking
                llm_generated_content.extend(
                    self._generate_chunks_concurrent_with_progress(
                        remaining_chunks, project_id, prompt_name, taxonomy_project, taxonomy_name,
                        datasets_per_chunk, concurrency, progress_callback,
                        processed_chunks, total_chunks, start_time, timeout_seconds
                    )
                )
            else:
                # Sequential processing with progress tracking
                for i, chunk in enumerate(remaining_chunks):
                    current_chunk_idx = processed_chunks + i

                    if progress_callback:
                        progress = 0.1 + (current_chunk_idx / total_chunks) * 0.8
                        progress_callback(progress, f"Processing chunk {current_chunk_idx + 1}/{total_chunks}...")

                    try:
                        qa_pairs = self._generate_qa_for_chunk(
                            project_id, prompt_name, chunk,
                            taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                            datasets_per_chunk=datasets_per_chunk
                        )
                        llm_generated_content.extend(qa_pairs)

                        # Update resume state
                        processed_chunks = current_chunk_idx + 1

                    except Exception as exc:
                        logger.warning(f"Chunk {current_chunk_idx + 1} generated an exception: {exc}")
                        continue

            if progress_callback:
                progress_callback(0.95, "Formatting dataset...")

            formatted_dataset = self.output_formatter.format_dataset(
                llm_generated_content, format_type
            )

            # Handle versioning if enabled
            if enable_versioning and self.version_manager and dataset_name:
                try:
                    version = self.version_manager.create_new_version(
                        project_id=project_id,
                        dataset_name=dataset_name,
                        entries=llm_generated_content,
                        description=f"Generated dataset with {len(llm_generated_content)} entries (resumable)",
                        metadata={
                            "prompt_name": prompt_name,
                            "format_type": format_type,
                            "concurrency": concurrency,
                            "taxonomy_project": taxonomy_project,
                            "taxonomy_name": taxonomy_name,
                            "generation_method": "qa_pairs_resumable"
                        }
                    )
                    logger.info(f"Created dataset version: {version}")
                except Exception as e:
                    logger.warning(f"Failed to create dataset version: {e}")

            if progress_callback:
                progress_callback(1.0, f"Dataset generation completed with {len(llm_generated_content)} items")

            # Trigger post-generation hooks (benchmarking, etc.)
            self._trigger_post_generation_hooks(project_id, llm_generated_content, {
                'prompt_name': 'qa_pairs',
                'format_type': format_type,
                'concurrency': concurrency,
                'taxonomy_project': None,  # Extract mode doesn't use taxonomy
                'taxonomy_name': None,     # Extract mode doesn't use taxonomy
                'generation_method': 'qa_pairs'
            })

        return formatted_dataset

    def _generate_from_extracted_with_progress(self, project_id: int, extracted_data: List[Dict[str, Any]],
                                             prompt_name: str, format_type: str, concurrency: int,
                                             taxonomy_project: Optional[str], taxonomy_name: Optional[str],
                                             datasets_per_chunk: int,
                                             progress_callback: Optional[Callable[[float, str], None]],
                                             start_time: float, timeout_seconds: Optional[float]) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """Generate from extracted data with progress tracking."""
        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            # Use concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                from functools import partial
                generate_func = partial(self._generate_qa_for_extracted_item, project_id, prompt_name,
                                       taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                       datasets_per_chunk=datasets_per_chunk)
                results = list(executor.map(generate_func, extracted_data))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            # Sequential processing with progress
            total_items = len(extracted_data)
            for i, item in enumerate(extracted_data):
                if progress_callback:
                    progress = 0.2 + (i / total_items) * 0.7
                    progress_callback(progress, f"Processing extracted item {i + 1}/{total_items}...")

                # Check timeout
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

                try:
                    qa_pairs = self._generate_qa_for_extracted_item(
                        project_id, prompt_name, item,
                        taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                        datasets_per_chunk=datasets_per_chunk
                    )
                    llm_generated_content.extend(qa_pairs)
                except Exception as exc:
                    logger.warning(f"Extracted item {item.get('chunk_id', 'unknown')} generated an exception: {exc}")

        formatted_dataset = self.output_formatter.format_dataset(
            llm_generated_content, format_type
        )

        # Trigger post-generation hooks (benchmarking, etc.)
        self._trigger_post_generation_hooks(project_id, llm_generated_content, {
            'prompt_name': 'qa_pairs',
            'format_type': format_type,
            'concurrency': concurrency,
            'taxonomy_project': taxonomy_project,
            'taxonomy_name': taxonomy_name,
            'generation_method': 'qa_pairs'
        })

        return formatted_dataset

    def _generate_chunks_concurrent_with_progress(self, chunks: List[str], project_id: int, prompt_name: str,
                                                taxonomy_project: Optional[str], taxonomy_name: Optional[str],
                                                datasets_per_chunk: int, concurrency: int,
                                                progress_callback: Optional[Callable[[float, str], None]],
                                                processed_chunks: int, total_chunks: int,
                                                start_time: float, timeout_seconds: Optional[float]) -> List[Dict[str, Any]]:
        """Generate chunks concurrently with progress tracking."""
        llm_generated_content: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all tasks
            future_to_chunk = {}
            for i, chunk in enumerate(chunks):
                future = executor.submit(
                    self._generate_qa_for_chunk, project_id, prompt_name, chunk,
                    taxonomy_project, taxonomy_name, datasets_per_chunk
                )
                future_to_chunk[future] = (i, chunk)

            # Process completed tasks with progress updates
            completed = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx, chunk = future_to_chunk[future]
                current_total_processed = processed_chunks + completed + 1

                try:
                    qa_pairs = future.result()
                    llm_generated_content.extend(qa_pairs)

                    if progress_callback:
                        progress = 0.1 + (current_total_processed / total_chunks) * 0.8
                        progress_callback(progress, f"Completed chunk {current_total_processed}/{total_chunks}")

                except Exception as exc:
                    logger.warning(f"Chunk {current_total_processed} generated an exception: {exc}")

                completed += 1

                # Check timeout
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

        return llm_generated_content

    def _generate_instruction_following_for_chunk(self, project_id: int, chunk: str,
                                                taxonomy_project: Optional[str] = None,
                                                taxonomy_name: Optional[str] = None,
                                                datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates instruction-response pairs for a single chunk.

        Args:
            project_id: The project ID.
            chunk: The text chunk to process.
            taxonomy_project: Optional taxonomy project.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Number of instruction-response pairs to generate per chunk.

        Returns:
            List of generated instruction-response pairs.
        """
        results = []

        # Define instruction types based on research
        instruction_types = [
            "explain", "describe", "analyze", "summarize",
            "create", "implement", "design", "troubleshoot"
        ]

        for i in range(datasets_per_chunk):
            # Select random instruction type for diversity
            import random
            instruction_type = random.choice(instruction_types)

            # Build prompt for instruction generation
            final_prompt = self._build_instruction_following_prompt(
                chunk, instruction_type, taxonomy_project, taxonomy_name
            )

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:1]:  # Take only first result per generation
                    if isinstance(item, dict) and 'instruction' in item and 'output' in item:
                        item["instruction_type"] = instruction_type
                        item["source_type"] = "instruction_following"
                        results.append(item)
            else:
                if isinstance(result, dict) and 'instruction' in result and 'output' in result:
                    result["instruction_type"] = instruction_type
                    result["source_type"] = "instruction_following"
                    results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def _build_instruction_following_prompt(self, chunk: str, instruction_type: str,
                                          taxonomy_project: Optional[str] = None,
                                          taxonomy_name: Optional[str] = None) -> str:
        """
        Builds a prompt for generating instruction-response pairs.

        Args:
            chunk: The source text chunk.
            instruction_type: Type of instruction to generate.
            taxonomy_project: Optional taxonomy project.
            taxonomy_name: Optional taxonomy name.

        Returns:
            Formatted prompt for LLM.
        """
        # Base prompt structure following Alpaca format
        base_prompt = f"""Generate an instruction-response pair based on the following content.

        Content: {chunk}

        Instruction type: {instruction_type}

CRITICAL CONSTRAINTS:
- ONLY use information and concepts that are explicitly present in the provided content above.
- DO NOT draw from external knowledge, general knowledge, or information not contained in the content.
- DO NOT generate synthetic or hypothetical content.
- If the content does not contain sufficient information for an instruction-response pair, do not generate one.
- Base all instructions and responses strictly on the content provided.

Return your response as a valid JSON object with exactly this structure:
{{
  "instruction": "A clear, actionable instruction related to the content",
  "input": "",
  "output": "A comprehensive, helpful response to the instruction"
}}

The instruction should be something someone would actually ask about this content.
The response should be detailed and directly answer the instruction.
"""

        # Add taxonomy context if available
        if taxonomy_name:
            base_prompt += f"\n\nContext: This content relates to the taxonomy category '{taxonomy_name}'."

        return base_prompt

    def _generate_with_custom_prompt(self, custom_prompt: str, chunk: str,
                                    taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                    datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates content using a custom prompt.

        Args:
            custom_prompt: The custom prompt text.
            chunk: The text chunk to process.
            taxonomy_project: Optional taxonomy project.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Number of items to generate per chunk.

        Returns:
            List of generated content items.
        """
        results = []
        for i in range(datasets_per_chunk):
            # Replace {chunk} placeholder with actual chunk text
            if "{chunk}" in custom_prompt:
                final_prompt = custom_prompt.replace("{chunk}", chunk)
            else:
                final_prompt = f"Content: {chunk}\n\n" + custom_prompt

            # Add clear JSON format instructions for the custom prompt
            json_instructions = f"""

IMPORTANT: You must respond with a valid JSON object (or array of objects) with exactly this structure:
{{
  "question": "The question you generated based on the content",
  "answer": "The answer to the question",
  "reasoning": "Brief explanation of your reasoning",
  "confidence_score": 0.8,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}}

CRITICAL CONSTRAINTS:
- ONLY use information explicitly present in the provided content.
- Generate questions that are directly answerable from the content provided.
- Base answers strictly on the content provided.
- If the content doesn't support generating meaningful questions and answers, return an empty array [].

Return ONLY the JSON object or array, no additional text or explanation."""

            if datasets_per_chunk > 1:
                json_instructions += f"\n\nGenerate {datasets_per_chunk} different clinical case question-answer pairs. Return them as a JSON array of objects."

            final_prompt += json_instructions

            result = self.llm_interaction.generate(final_prompt)
            logger.debug(f"LLM result for chunk {i+1}: {result} (type: {type(result)})")

            # Parse JSON string if needed
            if isinstance(result, str):
                try:
                    result = json.loads(result.strip())
                    logger.debug(f"Parsed JSON string to: {type(result)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    continue

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    if item and isinstance(item, dict):
                        results.append(item)
                        logger.debug(f"Added list item: {item.get('question', 'no question')[:50]}...")
                    else:
                        logger.warning(f"Skipping invalid list item: {item}")
            elif isinstance(result, dict) and result:
                results.append(result)
                logger.debug(f"Added dict result: {result.get('question', 'no question')[:50]}...")
            else:
                logger.warning(f"Skipping invalid result: {result}")

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def generate_dataset(
        self, project_id: int, chunks: Optional[List[str]] = None, prompt_name: str = "qa_pairs",
        format_type: str = "json", concurrency: int = 1,
        taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
        dataset_name: Optional[str] = None, enable_versioning: bool = False,
        datasets_per_chunk: int = 3, prefer_extracted: bool = True,
        only_validated: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generates a dataset from extracted data (preferred) or text chunks (fallback).

        Args:
            project_id: The ID of the project to associate with the prompt parameters.
            chunks: Optional list of text chunks to process (fallback when no extraction data).
            prompt_name: The name of the prompt to use.
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            taxonomy_project: Optional taxonomy project name.
            taxonomy_name: Optional taxonomy name.
            dataset_name: Optional name for the dataset (for versioning).
            enable_versioning: Whether to create a version record.
            datasets_per_chunk: Maximum number of datasets to generate per text chunk.
            prefer_extracted: Whether to prefer extracted data over raw chunks.
            only_validated: Whether to only include results that have passed validation.
            options: Optional Ollama API options to override defaults.

        Returns:
            The generated and formatted dataset.
        """
        # Try to use extracted data first if preferred
        if prefer_extracted:
            extracted_data = self.load_extracted_data(project_id, use_filesystem=self.enable_filesystem_storage,
                                                     load_chunk_text=False, only_validated=only_validated)
            if extracted_data:
                logger.info(f"Using {len(extracted_data)} extracted items for dataset generation")
                result = self.generate_from_extracted(
                    project_id=project_id,
                    extracted_data=extracted_data,
                    prompt_name=prompt_name,
                    format_type=format_type,
                    concurrency=concurrency,
                    taxonomy_project=taxonomy_project,
                    taxonomy_name=taxonomy_name,
                    datasets_per_chunk=datasets_per_chunk
                )

                # Handle versioning if enabled
                if enable_versioning and self.version_manager and dataset_name:
                    # Note: Versioning logic would need to be adapted for extracted data
                    # For now, we'll skip detailed versioning implementation
                    logger.info(f"Versioning requested for extracted data dataset: {dataset_name}")

                return result

        # Fallback to chunk-based generation
        if not chunks:
            raise ValueError("No chunks provided for dataset generation and no extracted data available.")

        logger.info(f"Using {len(chunks)} text chunks for dataset generation (fallback)")
        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            # Use concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Create partial function with fixed arguments
                from functools import partial
                generate_func = partial(self._generate_qa_for_chunk, project_id, prompt_name,
                                      taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                      datasets_per_chunk=datasets_per_chunk)
                # Map over chunks
                results = list(executor.map(generate_func, chunks))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            # Sequential processing
            for chunk in chunks:
                try:
                    qa_pairs = self._generate_qa_for_chunk(project_id, prompt_name, chunk,
                                                          taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                                          datasets_per_chunk=datasets_per_chunk)
                    llm_generated_content.extend(qa_pairs)
                except Exception as exc:
                    logger.error(f"Chunk generated an exception: {exc}")

        formatted_dataset = self.output_formatter.format_dataset(
            llm_generated_content, format_type
        )

        # Handle versioning if enabled
        if enable_versioning and self.version_manager and dataset_name:
            version = self.version_manager.create_new_version(
                project_id=project_id,
                dataset_name=dataset_name,
                entries=llm_generated_content,
                description=f"Generated dataset with {len(llm_generated_content)} entries",
                metadata={
                    "prompt_name": prompt_name,
                    "format_type": format_type,
                    "concurrency": concurrency,
                    "taxonomy_project": taxonomy_project,
                    "taxonomy_name": taxonomy_name,
                    "generation_method": "qa_pairs"
                }
            )
            logger.info(f"Created dataset version: {version}")

            # Record lineage information
            if self.version_manager:
                # Get the latest version record
                versions = self.version_manager.version_repo.get_versions_by_project(project_id, dataset_name)
                if versions:
                    version_record = versions[0]
                    self.version_manager.record_lineage(
                        dataset_version_id=version_record[0],
                        processing_parameters={
                            "concurrency": concurrency,
                            "format_type": format_type,
                            "prompt_name": prompt_name
                        }
                    )

        # Trigger post-generation hooks (benchmarking, etc.)
        self._trigger_post_generation_hooks(project_id, llm_generated_content, {
            'prompt_name': 'qa_pairs',
            'format_type': format_type,
            'concurrency': concurrency,
            'taxonomy_project': taxonomy_project,
            'taxonomy_name': taxonomy_name,
            'generation_method': 'qa_pairs'
        })

        return formatted_dataset

    def generate_questions_dataset(
        self, project_id: int, chunks: List[str], format_type: str, concurrency: int = 1,
        taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
        datasets_per_chunk: int = 3
    ) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generates a questions dataset from a list of text chunks.

        Args:
            project_id: The ID of the project.
            chunks: A list of text chunks to process.
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            taxonomy_project: Optional taxonomy project name.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Maximum number of datasets to generate per text chunk.

        Returns:
            The generated and formatted questions dataset.
        """
        if not chunks:
            raise ValueError("No chunks provided for questions generation.")

        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                from functools import partial
                generate_func = partial(self._generate_questions_for_chunk, project_id,
                                      taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                      datasets_per_chunk=datasets_per_chunk)
                results = list(executor.map(generate_func, chunks))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            for chunk in chunks:
                try:
                    results = self._generate_questions_for_chunk(project_id, chunk, taxonomy_project, taxonomy_name, datasets_per_chunk)
                    llm_generated_content.extend(results)
                except Exception as exc:
                    logger.error(f"Chunk generated an exception: {exc}")

        formatted_dataset = self.output_formatter.format_dataset(
            llm_generated_content, format_type
        )

        # Trigger post-generation hooks (benchmarking, etc.)
        self._trigger_post_generation_hooks(project_id, llm_generated_content, {
            'prompt_name': 'qa_pairs',
            'format_type': format_type,
            'concurrency': concurrency,
            'taxonomy_project': taxonomy_project,
            'taxonomy_name': taxonomy_name,
            'generation_method': 'qa_pairs'
        })

        return formatted_dataset

    def generate_answers_dataset(
        self, project_id: int, chunks: List[str], format_type: str, concurrency: int = 1,
        taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
        datasets_per_chunk: int = 3
    ) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generates an answers dataset from a list of text chunks.

        Args:
            project_id: The ID of the project.
            chunks: A list of text chunks to process.
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            taxonomy_project: Optional taxonomy project name.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Maximum number of datasets to generate per text chunk.

        Returns:
            The generated and formatted answers dataset.
        """
        if not chunks:
            raise ValueError("No chunks provided for answers generation.")

        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                from functools import partial
                generate_func = partial(self._generate_answers_for_chunk, project_id,
                                      taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                      datasets_per_chunk=datasets_per_chunk)
                results = list(executor.map(generate_func, chunks))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            for chunk in chunks:
                try:
                    results = self._generate_answers_for_chunk(project_id, chunk, taxonomy_project, taxonomy_name, datasets_per_chunk)
                    llm_generated_content.extend(results)
                except Exception as exc:
                    logger.error(f"Chunk generated an exception: {exc}")

        formatted_dataset = self.output_formatter.format_dataset(
            llm_generated_content, format_type
        )

        # Trigger post-generation hooks (benchmarking, etc.)
        self._trigger_post_generation_hooks(project_id, llm_generated_content, {
            'prompt_name': 'qa_pairs',
            'format_type': format_type,
            'concurrency': concurrency,
            'taxonomy_project': taxonomy_project,
            'taxonomy_name': taxonomy_name,
            'generation_method': 'qa_pairs'
        })

        return formatted_dataset

    def generate_summaries_dataset(
        self, project_id: int, chunks: List[str], format_type: str, concurrency: int = 1,
        taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
        datasets_per_chunk: int = 3
    ) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generates a summaries dataset from a list of text chunks.

        Args:
            project_id: The ID of the project.
            chunks: A list of text chunks to process.
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            taxonomy_project: Optional taxonomy project name.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Maximum number of datasets to generate per text chunk.

        Returns:
            The generated and formatted summaries dataset.
        """
        if not chunks:
            raise ValueError("No chunks provided for summaries generation.")

        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                from functools import partial
                generate_func = partial(self._generate_summary_for_chunk, project_id,
                                      taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                      datasets_per_chunk=datasets_per_chunk)
                results = list(executor.map(generate_func, chunks))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            for chunk in chunks:
                try:
                    results = self._generate_summary_for_chunk(project_id, chunk, taxonomy_project, taxonomy_name, datasets_per_chunk)
                    llm_generated_content.extend(results)
                except Exception as exc:
                    logger.error(f"Chunk generated an exception: {exc}")

        formatted_dataset = self.output_formatter.format_dataset(
            llm_generated_content, format_type
        )

        # Trigger post-generation hooks (benchmarking, etc.)
        self._trigger_post_generation_hooks(project_id, llm_generated_content, {
            'prompt_name': 'qa_pairs',
            'format_type': format_type,
            'concurrency': concurrency,
            'taxonomy_project': taxonomy_project,
            'taxonomy_name': taxonomy_name,
            'generation_method': 'qa_pairs'
        })

        return formatted_dataset

    def generate_evaluation_dataset(
        self,
        project_id: int,
        chunks: List[str],
        prompt_name: str,
        format_type: str,
        concurrency: int = 1,
        include_splits: bool = True,
        include_cv_folds: bool = True,
        include_adversarial: bool = True,
        include_difficulty_stratification: bool = True,
        datasets_per_chunk: int = 3,
        **kwargs
    ) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generates a comprehensive evaluation dataset with multiple splits and components.

        Args:
            project_id: The ID of the project to associate with the prompt parameters.
            chunks: A list of text chunks to process.
            prompt_name: The name of the prompt to use.
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            include_splits: Whether to include train/val/test splits.
            include_cv_folds: Whether to include cross-validation folds.
            include_adversarial: Whether to include adversarial examples.
            include_difficulty_stratification: Whether to include difficulty stratification.
            datasets_per_chunk: Maximum number of datasets to generate per text chunk.
            **kwargs: Additional parameters for evaluation dataset generation.

        Returns:
            The generated and formatted evaluation datasets.
        """
        if not chunks:
            raise ValueError("No chunks provided for evaluation dataset generation.")

        # Generate the base dataset
        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                from functools import partial
                generate_func = partial(self._generate_qa_for_chunk, project_id, prompt_name,
                                      datasets_per_chunk=datasets_per_chunk)
                results = list(executor.map(generate_func, chunks))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            for chunk in chunks:
                try:
                    qa_pairs = self._generate_qa_for_chunk(project_id, prompt_name, chunk,
                                                          datasets_per_chunk=datasets_per_chunk)
                    llm_generated_content.extend(qa_pairs)
                except Exception as exc:
                    logger.error(f"Chunk generated an exception: {exc}")

        # Create evaluation datasets using EvaluationDatasetGenerator
        eval_generator = EvaluationDatasetGenerator()
        evaluation_datasets = eval_generator.create_evaluation_datasets(
            dataset=llm_generated_content,
            include_splits=include_splits,
            include_cv_folds=include_cv_folds,
            include_adversarial=include_adversarial,
            include_difficulty_stratification=include_difficulty_stratification,
            llm_interaction=self.llm_interaction,
            **kwargs
        )

        # Format the evaluation datasets
        formatted_evaluation_datasets = self.output_formatter.format_dataset(
            evaluation_datasets, format_type
        )

        return formatted_evaluation_datasets

    def generate_instruction_following_dataset(
        self, project_id: int, chunks: List[str], format_type: str, concurrency: int = 1,
        taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
        datasets_per_chunk: int = 3
    ) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Generates an instruction-following dataset from a list of text chunks.

        Args:
            project_id: The ID of the project.
            chunks: A list of text chunks to process.
            format_type: The desired output format.
            concurrency: The number of parallel threads to use.
            taxonomy_project: Optional taxonomy project name.
            taxonomy_name: Optional taxonomy name.
            datasets_per_chunk: Maximum number of instruction-response pairs to generate per text chunk.

        Returns:
            The generated and formatted instruction-following dataset.
        """
        if not chunks:
            raise ValueError("No chunks provided for instruction-following generation.")

        llm_generated_content: List[Dict[str, Any]] = []

        if concurrency > 1:
            # Use concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                from functools import partial
                generate_func = partial(self._generate_instruction_following_for_chunk, project_id,
                                      taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                      datasets_per_chunk=datasets_per_chunk)
                results = list(executor.map(generate_func, chunks))
                for result_list in results:
                    llm_generated_content.extend(result_list)
        else:
            # Sequential processing
            for chunk in chunks:
                try:
                    results = self._generate_instruction_following_for_chunk(project_id, chunk,
                                                                             taxonomy_project, taxonomy_name,
                                                                             datasets_per_chunk)
                    llm_generated_content.extend(results)
                except Exception as exc:
                    logger.error(f"Chunk generated an exception: {exc}")

        formatted_dataset = self.output_formatter.format_dataset(
            llm_generated_content, format_type
        )

        # Trigger post-generation hooks (benchmarking, etc.)
        self._trigger_post_generation_hooks(project_id, llm_generated_content, {
            'prompt_name': 'qa_pairs',
            'format_type': format_type,
            'concurrency': concurrency,
            'taxonomy_project': taxonomy_project,
            'taxonomy_name': taxonomy_name,
            'generation_method': 'qa_pairs'
        })

        return formatted_dataset

    def _trigger_post_generation_hooks(self, project_id: Union[int, str], generated_content: List[Dict[str, Any]],
                                      generation_metadata: Dict[str, Any]) -> None:
        """
        Trigger post-generation hooks like automatic benchmarking.

        Args:
            project_id: The project ID
            generated_content: The generated dataset content
            generation_metadata: Metadata about the generation process
        """
        try:
            # Check if automatic benchmarking is enabled for this project
            from ...core.settings import backend_settings
            auto_benchmark = backend_settings.get_setting(f"project_{project_id}_auto_benchmark", False)

            if auto_benchmark and generated_content:
                # Trigger automatic benchmarking
                self._trigger_automatic_benchmarking(project_id, generated_content, generation_metadata)

        except Exception as e:
            logger.warning(f"Failed to trigger post-generation hooks: {e}")

    def _trigger_automatic_benchmarking(self, project_id: Union[int, str], generated_content: List[Dict[str, Any]],
                                       generation_metadata: Dict[str, Any]) -> None:
        """
        Trigger automatic benchmarking after dataset generation.

        Args:
            project_id: The project ID
            generated_content: The generated dataset content
            generation_metadata: Metadata about the generation process
        """
        try:
            from ...features.jobhandle.enhanced_job_queue import enhanced_job_queue_manager, JobType

            # Prepare benchmark job parameters
            benchmark_params = {
                "job_id": f"auto_benchmark_{project_id}_{generation_metadata.get('dataset_name', 'unknown')}",
                "project_id": project_id,
                "benchmark_suite": "glue",  # Default to GLUE for now
                "ai_config": {
                    "model_name": generation_metadata.get("model_name", "mistral:latest")
                },
                "benchmark_params": {
                    "auto_generated": True,
                    "source_dataset": generation_metadata.get("dataset_name"),
                    "generation_method": generation_metadata.get("generation_method")
                }
            }

            # Submit benchmark job
            job_id = enhanced_job_queue_manager.submit_job(
                job_type=JobType.BENCHMARKING,
                parameters=benchmark_params,
                priority=JobPriority.NORMAL,
                user_id=f"auto_benchmark_project_{project_id}"
            )

            logger.info(f"Triggered automatic benchmarking job {job_id} for project {project_id}")

        except Exception as e:
            logger.error(f"Failed to trigger automatic benchmarking: {e}")
