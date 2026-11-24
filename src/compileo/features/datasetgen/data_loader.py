from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json

from src.compileo.core.logging import get_logger
from src.compileo.storage.src.project.database_repositories import (
    ExtractionResultRepository, ExtractionJobRepository
)
from src.compileo.features.extraction.filesystem_storage import (
    FilesystemStorageManager, ExtractionFileManager
)

logger = get_logger(__name__)


class DataLoader:
    """
    Handles loading of extraction result data.
    
    Design principle:
    - Database is the source of truth for metadata (job_id, chunk_id, categories, file_path)
    - Filesystem contains the actual extraction data (extracted_entities, texts)
    - When loading, we get metadata from DB and content from filesystem
    """

    def __init__(
        self,
        extraction_result_repository: Optional[ExtractionResultRepository] = None,
        extraction_job_repository: Optional[ExtractionJobRepository] = None,
        filesystem_storage_manager: Optional[FilesystemStorageManager] = None,
        enable_filesystem_storage: bool = True
    ):
        self.extraction_result_repository = extraction_result_repository
        self.extraction_job_repository = extraction_job_repository
        
        # Initialize filesystem storage if needed
        if enable_filesystem_storage and not filesystem_storage_manager:
            file_manager = ExtractionFileManager()
            self.filesystem_storage_manager = FilesystemStorageManager(file_manager)
        else:
            self.filesystem_storage_manager = filesystem_storage_manager

    def load_extracted_data(
        self,
        project_id: str,
        owner_id: str,
        min_confidence: float = 0.0,
        load_chunk_text: bool = True,
        only_validated: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """Load extracted data - metadata from DB, content from filesystem."""
        if not self.extraction_result_repository:
            logger.error("No extraction result repository available")
            return None
        
        try:
            # Get metadata from database
            results = self.extraction_result_repository.get_results_by_project(
                project_id, owner_id, min_confidence=min_confidence
            )
            
            if not results:
                logger.warning(f"No extraction results found for project {project_id}")
                return None
            
            # Load full data from filesystem
            mapped_results = []
            for r in results:
                # Parse categories from database
                try:
                    categories = json.loads(r['categories']) if isinstance(r['categories'], str) else r['categories']
                except (json.JSONDecodeError, TypeError):
                    categories = []
                
                # Load extracted data from filesystem if file_path exists
                metadata = {}
                file_path = r.get('file_path')
                if file_path and self.filesystem_storage_manager:
                    try:
                        metadata = self._load_extraction_data_from_filesystem(file_path, owner_id)
                    except Exception as e:
                        logger.warning(f"Failed to load extraction data from {file_path}: {e}")
                
                # If filesystem load failed, try database fallback
                if not metadata and r.get('extracted_data'):
                    try:
                        metadata = json.loads(r['extracted_data']) if isinstance(r['extracted_data'], str) else r['extracted_data']
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                # Filter by validation status if requested
                if only_validated and not metadata.get('is_validated', False):
                    continue
                
                mapped_results.append({
                    'chunk_id': r.get('chunk_id'),
                    'categories': categories,
                    'confidence': r.get('confidence', 0.0),
                    'metadata': metadata
                })
            
            if not mapped_results:
                logger.warning(f"No extraction results passed validation filter for project {project_id}")
                return None
            
            logger.info(f"Loaded {len(mapped_results)} extraction results for project {project_id}")
            return mapped_results
            
        except Exception as e:
            logger.error(f"Failed to load extraction results: {e}")
            return None

    def load_extracted_data_by_job_id(
        self,
        job_id: str,
        owner_id: str,
        page: int = 1,
        page_size: int = 100,
        only_validated: bool = False
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Load extracted data for a specific job - metadata from DB, content from filesystem."""
        if not self.extraction_result_repository:
            logger.error("No extraction result repository available")
            return [], 0
        
        try:
            # Get metadata from database
            results, total = self.extraction_result_repository.get_results_by_job(
                job_id, owner_id, page=page, page_size=page_size
            )
            
            if not results:
                logger.warning(f"No extraction results found for job {job_id}")
                return [], 0
            
            # Load full data from filesystem
            mapped_results = []
            
            # Construct file path from job_id if not stored in database
            # File naming convention: storage/{owner_id}/extract/{project_id}/extraction_{job_id}_*.json
            constructed_file_path = None
            if results:
                project_id = results[0].get('project_id')
                if project_id:
                    constructed_file_path = f"extract/{project_id}/extraction_{job_id}"
                    logger.debug(f"DATA_LOADER_DEBUG: Constructed file path base: {constructed_file_path}")
            
            for r in results:
                # Parse categories from database
                try:
                    categories = json.loads(r['categories']) if isinstance(r['categories'], str) else r['categories']
                except (json.JSONDecodeError, TypeError):
                    categories = []
                
                # Load extracted data from filesystem
                metadata = {}
                file_path = r.get('file_path')
                
                # Try stored file_path first, then constructed path
                paths_to_try = []
                if file_path:
                    paths_to_try.append(file_path)
                if constructed_file_path:
                    paths_to_try.append(constructed_file_path)
                
                for path in paths_to_try:
                    if not path:
                        continue
                    try:
                        metadata = self._load_extraction_data_from_filesystem(path, owner_id)
                        if metadata:
                            logger.debug(f"DATA_LOADER_DEBUG: Loaded metadata from {path}, keys: {list(metadata.keys())}")
                            break
                    except Exception as e:
                        logger.debug(f"DATA_LOADER_DEBUG: Failed to load from {path}: {e}")
                        continue
                
                # If filesystem load failed, try database fallback
                if not metadata and r.get('extracted_data'):
                    try:
                        metadata = json.loads(r['extracted_data']) if isinstance(r['extracted_data'], str) else r['extracted_data']
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                # Filter by validation status if requested
                if only_validated and not metadata.get('is_validated', False):
                    continue
                
                # Validate that extracted_entities exists
                if 'extracted_entities' not in metadata:
                    logger.warning(f"Result {r.get('id', 'unknown')} missing extracted_entities. Keys: {list(metadata.keys())}")
                
                mapped_results.append({
                    'chunk_id': r.get('chunk_id'),
                    'categories': categories,
                    'confidence': r.get('confidence', 0.0),
                    'metadata': metadata
                })
            
            logger.info(f"Loaded {len(mapped_results)} extraction results for job {job_id} (total: {total})")
            return mapped_results, total
            
        except Exception as e:
            logger.error(f"Failed to load extraction results for job {job_id}: {e}")
            return [], 0

    def _load_extraction_data_from_filesystem(self, file_path: str, owner_id: str) -> Dict[str, Any]:
        """Load extraction data from filesystem file.
        
        Supports both exact paths and path patterns (for files with timestamps).
        """
        from src.compileo.storage.src.project.file_manager import FileManager
        import glob
        
        storage_base = FileManager.get_storage_base_path()
        
        # Handle both absolute and relative paths
        if Path(file_path).is_absolute():
            search_path = Path(file_path)
        else:
            search_path = storage_base / file_path
        
        # If exact path doesn't exist, try as pattern
        if not search_path.exists():
            # Try with owner_id prefix
            search_path = storage_base / owner_id / file_path
        
        if not search_path.exists():
            # Try glob pattern (for files with timestamps)
            pattern = str(search_path) + "*.json"
            matches = glob.glob(pattern)
            if not matches:
                # Try with owner_id prefix
                pattern = str(storage_base / owner_id / file_path) + "*.json"
                matches = glob.glob(pattern)
            
            if matches:
                # Use the most recent file
                full_path = Path(max(matches, key=lambda p: Path(p).stat().st_mtime))
                logger.debug(f"DATA_LOADER_DEBUG: Found matching file: {full_path.name}")
            else:
                raise FileNotFoundError(f"No extraction files found matching: {file_path}")
        else:
            full_path = search_path
        
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # The file contains results with metadata
        # Return combined metadata from all results
        if isinstance(data, dict) and 'results' in data:
            # New format: {metadata: {...}, results: [...]}
            all_entities = {}
            for result in data['results']:
                result_meta = result.get('metadata', {})
                if 'extracted_entities' in result_meta:
                    all_entities.update(result_meta['extracted_entities'])
            
            # Build combined metadata
            combined_metadata = data.get('metadata', {})
            if all_entities:
                combined_metadata['extracted_entities'] = all_entities
            return combined_metadata
        elif isinstance(data, list) and len(data) > 0:
            # Old format: list of results
            all_entities = {}
            for result in data:
                result_meta = result.get('metadata', {})
                if 'extracted_entities' in result_meta:
                    all_entities.update(result_meta['extracted_entities'])
            return {'extracted_entities': all_entities} if all_entities else {}
        
        return {}

    def _enrich_with_chunk_text(self, extracted_items: List[Dict[str, Any]], owner_id: str):
        """Enrich extracted items with full chunk text from filesystem."""
        # Note: Chunk text enrichment should be done via the ChunkRepository
        # This is a placeholder for future implementation
        pass

    def load_extracted_data_batch(
        self,
        project_id: str,
        owner_id: str,
        page: int = 1,
        page_size: int = 50,
        only_validated: bool = False
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Load a batch of extracted data for a project."""
        if not self.extraction_result_repository:
            return [], False

        offset = (page - 1) * page_size
        
        try:
            results = self.extraction_result_repository.get_results_by_project(
                project_id, owner_id, limit=page_size + 1, offset=offset
            )
            
            has_more = len(results) > page_size
            batch = results[:page_size]
            
            mapped_batch = []
            for r in batch:
                try:
                    categories = json.loads(r['categories']) if isinstance(r['categories'], str) else r['categories']
                except (json.JSONDecodeError, TypeError):
                    categories = []
                
                # Load from filesystem
                metadata = {}
                file_path = r.get('file_path')
                if file_path and self.filesystem_storage_manager:
                    try:
                        metadata = self._load_extraction_data_from_filesystem(file_path, owner_id)
                    except Exception:
                        pass
                
                # Fallback to database
                if not metadata and r.get('extracted_data'):
                    try:
                        metadata = json.loads(r['extracted_data']) if isinstance(r['extracted_data'], str) else r['extracted_data']
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                if only_validated and not metadata.get('is_validated', False):
                    continue
                
                mapped_batch.append({
                    'chunk_id': r.get('chunk_id'),
                    'categories': categories,
                    'confidence': r.get('confidence', 0.0),
                    'metadata': metadata
                })
            
            self._enrich_with_chunk_text(mapped_batch, owner_id)
            return mapped_batch, has_more
            
        except Exception as e:
            logger.error(f"Failed to load extraction batch: {e}")
            return [], False
