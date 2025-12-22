from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, Iterator
from datetime import datetime, timedelta
import threading
import time

from .models import ExtractionResult, ExtractionResultChunk, ExtractionResultMetadata
from ...core.logging import get_logger

logger = get_logger(__name__)


class ResultChunkingStrategy(ABC):
    """Abstract base class for result chunking strategies."""

    @abstractmethod
    def chunk_results(self, results: List[ExtractionResult], chunk_size: int) -> List[ExtractionResultChunk]:
        """Chunk results into manageable pieces."""
        pass

    @abstractmethod
    def get_chunk_metadata(self, chunk: ExtractionResultChunk) -> ExtractionResultMetadata:
        """Get metadata for a result chunk."""
        pass


class TimeBasedChunkingStrategy(ResultChunkingStrategy):
    """Chunk results based on time windows."""

    def chunk_results(self, results: List[ExtractionResult], chunk_size: int) -> List[ExtractionResultChunk]:
        chunks = []
        current_chunk = []
        current_chunk_id: Optional[str] = None

        for result in sorted(results, key=lambda x: x.created_at):
            chunk_id = result.created_at.strftime("%Y%m%d_%H")

            if current_chunk_id != chunk_id or len(current_chunk) >= chunk_size:
                if current_chunk and current_chunk_id:
                    chunks.append(ExtractionResultChunk(
                        chunk_id=current_chunk_id,
                        results=current_chunk
                    ))
                current_chunk = []
                current_chunk_id = chunk_id

            current_chunk.append(result)

        if current_chunk and current_chunk_id:
            chunks.append(ExtractionResultChunk(
                chunk_id=current_chunk_id,
                results=current_chunk
            ))

        return chunks

    def get_chunk_metadata(self, chunk: ExtractionResultChunk) -> ExtractionResultMetadata:
        if not chunk.results:
            return ExtractionResultMetadata(
                total_results=0,
                categories=[],
                size_bytes=0,
                created_at=datetime.utcnow()
            )

        categories = set()
        total_size = 0

        for result in chunk.results:
            categories.update(result.categories)
            # Estimate size (rough calculation)
            total_size += len(str(result.extracted_data)) + len(result.chunk_id)

        return ExtractionResultMetadata(
            total_results=len(chunk.results),
            categories=list(categories),
            size_bytes=total_size,
            created_at=min(r.created_at for r in chunk.results)
        )


class SizeBasedChunkingStrategy(ResultChunkingStrategy):
    """Chunk results based on result count."""

    def chunk_results(self, results: List[ExtractionResult], chunk_size: int) -> List[ExtractionResultChunk]:
        chunks = []
        current_chunk = []
        chunk_counter = 0

        for result in results:
            if len(current_chunk) >= chunk_size:
                chunk_id = f"size_chunk_{chunk_counter}"
                chunks.append(ExtractionResultChunk(
                    chunk_id=chunk_id,
                    results=current_chunk
                ))
                current_chunk = []
                chunk_counter += 1

            current_chunk.append(result)

        if current_chunk:
            chunk_id = f"size_chunk_{chunk_counter}"
            chunks.append(ExtractionResultChunk(
                chunk_id=chunk_id,
                results=current_chunk
            ))

        return chunks

    def get_chunk_metadata(self, chunk: ExtractionResultChunk) -> ExtractionResultMetadata:
        if not chunk.results:
            return ExtractionResultMetadata(
                total_results=0,
                categories=[],
                size_bytes=0,
                created_at=datetime.utcnow()
            )

        categories = set()
        total_size = 0

        for result in chunk.results:
            categories.update(result.categories)
            # Estimate size (rough calculation)
            total_size += len(str(result.extracted_data)) + len(result.chunk_id)

        return ExtractionResultMetadata(
            total_results=len(chunk.results),
            categories=list(categories),
            size_bytes=total_size,
            created_at=min(r.created_at for r in chunk.results)
        )


class ResultIndexingStrategy(ABC):
    """Abstract base class for result indexing strategies."""

    @abstractmethod
    def index_results(self, results: List[ExtractionResult]) -> Dict[str, Any]:
        """Create searchable index for results."""
        pass

    @abstractmethod
    def search_index(self, index: Dict[str, Any], query: Dict[str, Any]) -> List[str]:
        """Search the index for matching result IDs."""
        pass


class CategoryBasedIndexingStrategy(ResultIndexingStrategy):
    """Index results by categories for fast lookup."""

    def index_results(self, results: List[ExtractionResult]) -> Dict[str, Any]:
        index = {
            'categories': {},
            'chunk_ids': {},
            'created_at': []
        }

        for result in results:
            # Index by categories
            for category in result.categories:
                if category not in index['categories']:
                    index['categories'][category] = []
                index['categories'][category].append(result.id)

            # Index by chunk_id
            if result.chunk_id not in index['chunk_ids']:
                index['chunk_ids'][result.chunk_id] = []
            index['chunk_ids'][result.chunk_id].append(result.id)

            # Keep track of creation times
            index['created_at'].append((result.created_at, result.id))

        # Sort creation times
        index['created_at'].sort(key=lambda x: x[0])

        return index

    def search_index(self, index: Dict[str, Any], query: Dict[str, Any]) -> List[str]:
        result_ids = set()

        # Search by categories
        if 'categories' in query:
            for category in query['categories']:
                if category in index['categories']:
                    result_ids.update(index['categories'][category])

        # Search by chunk_id
        if 'chunk_id' in query:
            if query['chunk_id'] in index['chunk_ids']:
                result_ids.update(index['chunk_ids'][query['chunk_id']])

        # Filter by date range
        if 'date_from' in query or 'date_to' in query:
            filtered_ids = set()
            for created_at, result_id in index['created_at']:
                if 'date_from' in query and created_at < query['date_from']:
                    continue
                if 'date_to' in query and created_at > query['date_to']:
                    continue
                filtered_ids.add(result_id)
            result_ids = result_ids.intersection(filtered_ids) if result_ids else filtered_ids

        return list(result_ids)


class FullTextIndexingStrategy(ResultIndexingStrategy):
    """Index results for full-text search across extracted data."""

    def index_results(self, results: List[ExtractionResult]) -> Dict[str, Any]:
        index = {
            'full_text': {},  # word -> [result_ids]
            'result_texts': {},  # result_id -> searchable text
            'metadata': {}  # result_id -> metadata
        }

        for result in results:
            result_id = str(result.id)
            searchable_text = self._extract_searchable_text(result)

            # Store the searchable text
            index['result_texts'][result_id] = searchable_text

            # Store metadata
            index['metadata'][result_id] = {
                'confidence': result.confidence,
                'categories': result.categories,
                'created_at': result.created_at,
                'chunk_id': result.chunk_id
            }

            # Index individual words
            words = self._tokenize_text(searchable_text)
            for word in words:
                if word not in index['full_text']:
                    index['full_text'][word] = []
                if result_id not in index['full_text'][word]:
                    index['full_text'][word].append(result_id)

        return index

    def search_index(self, index: Dict[str, Any], query: Dict[str, Any]) -> List[str]:
        result_ids = set()

        # Full-text search
        if 'full_text' in query:
            search_terms = self._tokenize_text(query['full_text'])
            term_results = []

            for term in search_terms:
                if term in index['full_text']:
                    term_results.append(set(index['full_text'][term]))

            if term_results:
                # AND operation - result must contain all terms
                result_ids = set.intersection(*term_results)

        # Filter by metadata
        if result_ids:
            filtered_ids = set()
            for result_id in result_ids:
                metadata = index['metadata'].get(result_id, {})

                # Filter by confidence threshold
                if 'min_confidence' in query:
                    if metadata.get('confidence', 0) < query['min_confidence']:
                        continue

                # Filter by categories
                if 'categories' in query:
                    result_categories = set(metadata.get('categories', []))
                    query_categories = set(query['categories'])
                    if not query_categories.issubset(result_categories):
                        continue

                # Filter by date range
                if 'date_from' in query or 'date_to' in query:
                    created_at = metadata.get('created_at')
                    if created_at:
                        if 'date_from' in query and created_at < query['date_from']:
                            continue
                        if 'date_to' in query and created_at > query['date_to']:
                            continue

                filtered_ids.add(result_id)

            result_ids = filtered_ids

        return list(result_ids)

    def _extract_searchable_text(self, result: ExtractionResult) -> str:
        """Extract searchable text from a result."""
        texts = []

        # Add categories
        texts.extend(result.categories)

        # Add extracted data values
        def extract_text(obj):
            if isinstance(obj, str):
                return obj
            elif isinstance(obj, dict):
                return ' '.join(extract_text(v) for v in obj.values())
            elif isinstance(obj, list):
                return ' '.join(extract_text(item) for item in obj)
            else:
                return str(obj)

        texts.append(extract_text(result.extracted_data))

        return ' '.join(texts).lower()

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization - split on whitespace and remove punctuation."""
        import re
        # Remove punctuation and split on whitespace
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove common stop words (simple implementation)
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                     'to', 'was', 'will', 'with'}
        return [word for word in words if word not in stop_words and len(word) > 2]


class ResultCacheInterface(Protocol):
    """Protocol for result caching implementations."""

    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL in seconds."""
        ...

    def delete(self, key: str) -> bool:
        """Delete cached value by key."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...


class MemoryResultCache:
    """In-memory result cache implementation."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry:
            expires_at = entry.get('expires_at')
            if expires_at and datetime.utcnow() > expires_at:
                del self._cache[key]
                return None
            return entry['value']
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = None
        if ttl:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        self._cache[key] = {
            'value': value,
            'expires_at': expires_at
        }

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()


class ResultRetentionPolicy(ABC):
    """Abstract base class for result retention policies."""

    @abstractmethod
    def should_retain(self, result: ExtractionResult, current_time: datetime) -> bool:
        """Determine if a result should be retained."""
        pass

    @abstractmethod
    def get_cleanup_candidates(self, results: List[ExtractionResult], current_time: datetime) -> List[ExtractionResult]:
        """Get results that should be cleaned up."""
        pass


class TimeBasedRetentionPolicy(ResultRetentionPolicy):
    """Retain results for a specified time period."""

    def __init__(self, retention_days: int = 30):
        self.retention_period = timedelta(days=retention_days)

    def should_retain(self, result: ExtractionResult, current_time: datetime) -> bool:
        return current_time - result.created_at <= self.retention_period

    def get_cleanup_candidates(self, results: List[ExtractionResult], current_time: datetime) -> List[ExtractionResult]:
        return [r for r in results if not self.should_retain(r, current_time)]


class SizeBasedRetentionPolicy(ResultRetentionPolicy):
    """Retain results up to a maximum total size."""

    def __init__(self, max_size_bytes: int = 100 * 1024 * 1024):  # 100MB default
        self.max_size_bytes = max_size_bytes

    def should_retain(self, result: ExtractionResult, current_time: datetime) -> bool:
        # This would need context of total size, simplified for now
        return True

    def get_cleanup_candidates(self, results: List[ExtractionResult], current_time: datetime) -> List[ExtractionResult]:
        # Sort by creation time, oldest first
        sorted_results = sorted(results, key=lambda x: x.created_at)

        total_size = 0
        candidates = []

        for result in sorted_results:
            result_size = len(str(result.extracted_data)) + len(result.chunk_id)
            if total_size + result_size > self.max_size_bytes:
                candidates.append(result)
            else:
                total_size += result_size

        return candidates


class ExtractionResultStorageManager:
    """Comprehensive manager for extraction result storage, chunking, indexing, and cleanup."""

    def __init__(
        self,
        chunking_strategy: ResultChunkingStrategy,
        indexing_strategy: ResultIndexingStrategy,
        retention_policy: ResultRetentionPolicy,
        cache: ResultCacheInterface,
        db_connection=None
    ):
        self.chunking_strategy = chunking_strategy
        self.indexing_strategy = indexing_strategy
        self.retention_policy = retention_policy
        self.cache = cache
        self.db_connection = db_connection

        # Background cleanup thread
        self.cleanup_thread = None
        self.cleanup_interval = 3600  # 1 hour
        self.running = False

    def store_results(self, results: List[ExtractionResult]) -> List[str]:
        """Store results with chunking and indexing."""
        if not results:
            return []

        # Chunk the results
        chunks = self.chunking_strategy.chunk_results(results, chunk_size=100)

        stored_chunk_ids = []

        for chunk in chunks:
            chunk_id = chunk.chunk_id

            # Store chunk metadata in cache
            metadata = self.chunking_strategy.get_chunk_metadata(chunk)
            self.cache.set(f"chunk_metadata:{chunk_id}", metadata.model_dump(), ttl=3600)

            # Store individual results
            for result in chunk.results:
                self._store_single_result(result)

            stored_chunk_ids.append(chunk_id)

        # Update global index
        self._update_global_index(results)

        return stored_chunk_ids

    def _store_single_result(self, result: ExtractionResult) -> None:
        """Store a single result in the database."""
        if self.db_connection:
            # Use database repository
            from ...storage.src.project.database_repositories import ExtractionResultRepository
            repo = ExtractionResultRepository(self.db_connection)
            repo.create_result(
                job_id=str(result.job_id),  # Ensure string
                chunk_id=result.chunk_id,
                categories=result.categories,
                confidence=result.confidence,
                extracted_data={},  # Empty dict to avoid redundancy with filesystem data
                project_id=str(result.project_id) if result.project_id else None,
                id=str(result.id) if result.id else None,
                created_at=result.created_at,
                file_path=None  # Can be populated if integrated with filesystem storage
            )
        else:
            # Fallback to cache-only storage
            cache_key = f"result:{result.id}"
            self.cache.set(cache_key, result.model_dump(), ttl=86400)  # 24 hours

    def _update_global_index(self, results: List[ExtractionResult]) -> None:
        """Update the global search index with new results."""
        # Get existing index or create new one
        existing_index = self.cache.get("global_index") or {
            'categories': {},
            'chunk_ids': {},
            'created_at': []
        }

        # Update with new results
        for result in results:
            # Index by categories
            for category in result.categories:
                if category not in existing_index['categories']:
                    existing_index['categories'][category] = []
                if result.id not in existing_index['categories'][category]:
                    existing_index['categories'][category].append(result.id)

            # Index by chunk_id
            if result.chunk_id not in existing_index['chunk_ids']:
                existing_index['chunk_ids'][result.chunk_id] = []
            if result.id not in existing_index['chunk_ids'][result.chunk_id]:
                existing_index['chunk_ids'][result.chunk_id].append(result.id)

            # Keep track of creation times
            existing_index['created_at'].append((result.created_at, result.id))

        # Sort creation times and deduplicate
        existing_index['created_at'] = list(set(existing_index['created_at']))
        existing_index['created_at'].sort(key=lambda x: x[0])

        # Store updated index
        self.cache.set("global_index", existing_index, ttl=None)  # Persistent

    def search_results(self, query: Dict[str, Any], limit: int = 50, offset: int = 0) -> List[ExtractionResult]:
        """Search results with filtering and pagination."""
        # Get global index
        global_index = self.cache.get("global_index")
        if not global_index:
            return []

        # Use indexing strategy for search
        result_ids = self.indexing_strategy.search_index(global_index, query)

        # Apply pagination
        paginated_ids = result_ids[offset:offset + limit]

        # Retrieve results
        results = []
        for result_id in paginated_ids:
            result_data = self.cache.get(f"result:{result_id}")
            if result_data:
                results.append(ExtractionResult(**result_data))

        return results

    def get_results_by_chunk(self, chunk_id: str, lazy: bool = False) -> Iterator[ExtractionResult]:
        """Get results for a specific chunk with optional lazy loading."""
        results = []
        if self.db_connection:
            from ...storage.src.project.database_repositories import ExtractionResultRepository
            repo = ExtractionResultRepository(self.db_connection)
            db_results = repo.get_results_by_chunk(chunk_id)

            for row in db_results:
                import json
                categories = row.get('categories', '[]')
                if isinstance(categories, str):
                    try:
                        categories = json.loads(categories)
                    except:
                        categories = []
                
                extracted_data = row.get('extracted_data', '{}')
                if isinstance(extracted_data, str):
                    try:
                        extracted_data = json.loads(extracted_data)
                    except:
                        extracted_data = {}

                result = ExtractionResult(
                    id=row.get('id'),
                    job_id=row.get('job_id'),
                    project_id=row.get('project_id'),
                    chunk_id=row.get('chunk_id'),
                    categories=categories,
                    confidence=row.get('confidence', 0.0),
                    extracted_data=extracted_data,
                    created_at=row.get('created_at')
                )
                if lazy:
                    yield result
                else:
                    results.append(result)
        else:
            # Cache-based retrieval (simplified)
            # In practice, you'd need to maintain chunk-to-result mappings
            pass

        if not lazy:
            return iter(results)

    def get_chunk_metadata(self, chunk_id: str) -> Optional[ExtractionResultMetadata]:
        """Get metadata for a chunk."""
        cached_metadata = self.cache.get(f"chunk_metadata:{chunk_id}")
        if cached_metadata:
            return ExtractionResultMetadata(**cached_metadata)
        return None

    def cleanup_old_results(self, force: bool = False) -> int:
        """Clean up old results based on retention policy."""
        current_time = datetime.utcnow()
        cleaned_count = 0

        if self.db_connection:
            from ...storage.src.project.database_repositories import ExtractionResultRepository
            repo = ExtractionResultRepository(self.db_connection)

            # Get all results (in practice, you'd paginate this)
            # This is a simplified implementation
            all_results = []  # You'd need to implement a method to get all results

            cleanup_candidates = self.retention_policy.get_cleanup_candidates(all_results, current_time)

            for result in cleanup_candidates:
                # Delete from database
                # repo.delete_result(result.id)  # You'd need to implement this
                # Remove from cache
                self.cache.delete(f"result:{result.id}")
                cleaned_count += 1

        return cleaned_count

    def start_background_cleanup(self) -> None:
        """Start background cleanup thread."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return

        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def stop_background_cleanup(self) -> None:
        """Stop background cleanup thread."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)

    def _cleanup_worker(self) -> None:
        """Background cleanup worker."""
        while self.running:
            try:
                self.cleanup_old_results()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            time.sleep(self.cleanup_interval)

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        global_index = self.cache.get("global_index") or {
            'categories': {},
            'chunk_ids': {},
            'created_at': []
        }

        return {
            'total_results': len(global_index['created_at']),
            'total_chunks': len(global_index['chunk_ids']),
            'total_categories': len(global_index['categories']),
            'oldest_result': global_index['created_at'][0][0] if global_index['created_at'] else None,
            'newest_result': global_index['created_at'][-1][0] if global_index['created_at'] else None
        }