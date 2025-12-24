"""
Enhanced result search with pagination, lazy loading, and performance optimizations.
"""

from typing import List, Dict, Any, Optional, Tuple, Iterator
from datetime import datetime
import json
from functools import lru_cache


class LazyResultIterator:
    """Iterator for lazy loading of large result sets."""

    def __init__(self, db_connection, query_params: Dict[str, Any], page_size: int = 100):
        self.db_connection = db_connection
        self.query_params = query_params
        self.page_size = page_size
        self.current_page = 0
        self.cache: Dict[int, List[Dict[str, Any]]] = {}
        self.total_count = None
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration

        # Get next page
        page_data = self._get_page(self.current_page)
        if not page_data:
            self._exhausted = True
            raise StopIteration

        # Return individual results from page
        if hasattr(self, '_page_index'):
            self._page_index += 1
            if self._page_index >= len(page_data):
                self._page_index = 0
                self.current_page += 1
                if self.current_page * self.page_size >= (self.total_count or float('inf')):
                    self._exhausted = True
                    raise StopIteration
                page_data = self._get_page(self.current_page)
                if not page_data:
                    self._exhausted = True
                    raise StopIteration
            return self._convert_to_result(page_data[self._page_index])
        else:
            self._page_index = 0
            return self._convert_to_result(page_data[self._page_index])

    def _get_page(self, page_num: int) -> List[Dict[str, Any]]:
        """Get a page of results."""
        if page_num in self.cache:
            return self.cache[page_num]

        offset = page_num * self.page_size
        results, total = self._execute_query(limit=self.page_size, offset=offset)

        if self.total_count is None:
            self.total_count = total

        self.cache[page_num] = results
        return results

    def _execute_query(self, limit: int, offset: int) -> Tuple[List[Dict[str, Any]], int]:
        """Execute the search query with pagination."""
        from .result_search import ResultSearchEngine
        search_engine = ResultSearchEngine(self.db_connection)

        # Extract parameters
        query = self.query_params.get('query', '')
        categories = self.query_params.get('categories')
        min_confidence = self.query_params.get('min_confidence')
        date_from = self.query_params.get('date_from')
        date_to = self.query_params.get('date_to')

        return search_engine.full_text_search(
            query=query,
            categories=categories,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset
        )

    def _convert_to_result(self, row_data: Dict[str, Any]):
        """Convert database row to ExtractionResult."""
        from .models import ExtractionResult
        # Note: extracted_data might be empty if loaded from database only
        # Full data should be retrieved from filesystem if needed
        return ExtractionResult(
            id=row_data.get('id'),
            job_id=row_data.get('job_id'),
            chunk_id=row_data.get('chunk_id'),
            categories=json.loads(row_data.get('categories', '[]')),
            confidence=row_data.get('confidence', 0.0),
            extracted_data=json.loads(row_data.get('extracted_data', '{}')),
            created_at=row_data.get('created_at')
        )

    def get_total_count(self) -> int:
        """Get total count without loading all results."""
        if self.total_count is None:
            # Execute a count-only query
            _, total = self._execute_query(limit=1, offset=0)
            self.total_count = total
        return self.total_count

    def get_page(self, page_num: int) -> List[Any]:
        """Get a specific page of results."""
        return [self._convert_to_result(row) for row in self._get_page(page_num)]

    def clear_cache(self):
        """Clear the iterator cache."""
        self.cache.clear()


class PaginatedResultSearch:
    """Search engine with advanced pagination and caching."""

    def __init__(self, db_connection, cache=None, page_size: int = 50):
        self.db_connection = db_connection
        self.cache = cache
        self.page_size = page_size
        self._query_cache: Dict[str, Tuple[List[Any], int, datetime]] = {}

    def search_with_pagination(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        page: int = 0,
        per_page: Optional[int] = None,
        use_cache: bool = True
    ) -> Tuple[List[Any], int, Dict[str, Any]]:
        """
        Search with pagination and metadata.

        Returns:
            Tuple of (results, total_count, metadata)
        """
        per_page = per_page or self.page_size
        offset = page * per_page

        # Create cache key
        cache_key = self._create_cache_key(
            query, categories, min_confidence, date_from, date_to, offset, per_page
        )

        # Check cache
        if use_cache and self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

        # Execute search
        from .result_search import ResultSearchEngine
        search_engine = ResultSearchEngine(self.db_connection)

        results, total_count = search_engine.full_text_search(
            query=query,
            categories=categories,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            limit=per_page,
            offset=offset
        )

        # Create metadata
        metadata = {
            'page': page,
            'per_page': per_page,
            'total_pages': (total_count + per_page - 1) // per_page,
            'has_next': (page + 1) * per_page < total_count,
            'has_prev': page > 0,
            'total_count': total_count,
            'cached': False
        }

        result_tuple = (results, total_count, metadata)

        # Cache result
        if use_cache and self.cache:
            self.cache.set(cache_key, result_tuple, ttl=300)  # 5 minutes
            metadata['cached'] = True

        return result_tuple

    def get_lazy_iterator(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        page_size: Optional[int] = None
    ) -> LazyResultIterator:
        """Get a lazy iterator for large result sets."""
        query_params = {
            'query': query,
            'categories': categories,
            'min_confidence': min_confidence,
            'date_from': date_from,
            'date_to': date_to
        }

        return LazyResultIterator(
            self.db_connection,
            query_params,
            page_size=page_size or self.page_size
        )

    def get_category_stats_paginated(
        self,
        project_id: Optional[int] = None,
        page: int = 0,
        per_page: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get category statistics with pagination."""
        from .result_search import ResultSearchEngine
        search_engine = ResultSearchEngine(self.db_connection)

        # Get all stats
        all_stats = search_engine.get_category_stats(project_id)

        # Paginate
        categories = list(all_stats['categories'].items())
        total_count = len(categories)

        start_idx = page * per_page
        end_idx = start_idx + per_page

        paginated_categories = categories[start_idx:end_idx]

        return paginated_categories, total_count

    def get_chunk_stats_paginated(
        self,
        project_id: Optional[int] = None,
        page: int = 0,
        per_page: int = 20,
        sort_by: str = 'oldest_result',
        sort_desc: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get chunk statistics with pagination and sorting."""
        from .result_search import ResultSearchEngine
        search_engine = ResultSearchEngine(self.db_connection)

        # Get all stats
        all_stats = search_engine.get_chunk_stats(project_id)

        # Sort chunks
        chunks = list(all_stats['chunks'].values())
        reverse_sort = sort_desc

        if sort_by == 'oldest_result':
            chunks.sort(key=lambda x: x['oldest_result'], reverse=reverse_sort)
        elif sort_by == 'result_count':
            chunks.sort(key=lambda x: x['result_count'], reverse=reverse_sort)
        elif sort_by == 'avg_confidence':
            chunks.sort(key=lambda x: x['avg_confidence'], reverse=reverse_sort)

        # Paginate
        total_count = len(chunks)
        start_idx = page * per_page
        end_idx = start_idx + per_page

        paginated_chunks = chunks[start_idx:end_idx]

        return paginated_chunks, total_count

    def _create_cache_key(self, *args) -> str:
        """Create a cache key from search parameters."""
        # Convert all args to strings and hash
        key_parts = []
        for arg in args:
            if arg is None:
                key_parts.append('None')
            elif isinstance(arg, (list, tuple)):
                key_parts.append(','.join(str(x) for x in arg))
            elif isinstance(arg, datetime):
                key_parts.append(arg.isoformat())
            else:
                key_parts.append(str(arg))

        return f"search_{'_'.join(key_parts)}"

    def clear_cache(self):
        """Clear all cached search results."""
        if self.cache:
            # Remove all search-related cache entries
            # This is a simplified implementation
            pass

    def get_search_suggestions(
        self,
        partial_query: str,
        limit: int = 10
    ) -> List[str]:
        """Get search suggestions based on partial query."""
        cursor = self.db_connection.cursor()

        # Get categories that match
        cursor.execute("""
            SELECT DISTINCT jsonb_array_elements_text(COALESCE(categories, '[]'::jsonb)) as category
            FROM extraction_results
            WHERE jsonb_array_elements_text(COALESCE(categories, '[]'::jsonb)) ILIKE ?
            LIMIT ?
        """, (f"{partial_query}%", limit))

        categories = [row[0] for row in cursor.fetchall()]

        # Get common words from extracted data (simplified)
        cursor.execute("""
            SELECT DISTINCT word
            FROM (
                SELECT unnest(string_to_array(lower(extracted_data::text), ' ')) as word
                FROM extraction_results
                WHERE extracted_data::text ILIKE ?
                LIMIT 100
            ) words
            WHERE length(word) > 3
            LIMIT ?
        """, (f"%{partial_query}%", limit))

        words = [row[0] for row in cursor.fetchall()]

        # Combine and deduplicate
        suggestions = list(set(categories + words))
        return suggestions[:limit]


class ResultBatchProcessor:
    """Process large result sets in batches for memory efficiency."""

    def __init__(self, db_connection, batch_size: int = 1000):
        self.db_connection = db_connection
        self.batch_size = batch_size

    def process_in_batches(
        self,
        query_params: Dict[str, Any],
        processor_func,
        batch_callback=None
    ) -> Dict[str, Any]:
        """
        Process results in batches to avoid memory issues.

        Args:
            query_params: Search parameters
            processor_func: Function to process each batch
            batch_callback: Optional callback after each batch

        Returns:
            Processing statistics
        """
        lazy_iterator = LazyResultIterator(self.db_connection, query_params, self.batch_size)

        stats = {
            'total_processed': 0,
            'batches_processed': 0,
            'errors': 0,
            'start_time': datetime.utcnow()
        }

        try:
            batch = []
            for result in lazy_iterator:
                batch.append(result)

                if len(batch) >= self.batch_size:
                    # Process batch
                    batch_result = processor_func(batch)
                    stats['total_processed'] += len(batch)
                    stats['batches_processed'] += 1

                    if batch_callback:
                        batch_callback(stats, batch_result)

                    batch = []

            # Process remaining items
            if batch:
                batch_result = processor_func(batch)
                stats['total_processed'] += len(batch)
                stats['batches_processed'] += 1

                if batch_callback:
                    batch_callback(stats, batch_result)

        except Exception as e:
            stats['errors'] += 1
            stats['error_message'] = str(e)

        stats['end_time'] = datetime.utcnow()
        stats['duration_seconds'] = (stats['end_time'] - stats['start_time']).total_seconds()

        return stats

    def export_to_file(
        self,
        query_params: Dict[str, Any],
        output_file: str,
        format: str = 'jsonl'
    ) -> Dict[str, Any]:
        """Export large result sets to file without loading everything in memory."""
        import json

        def write_batch(batch):
            with open(output_file, 'a', encoding='utf-8') as f:
                for result in batch:
                    if format == 'jsonl':
                        f.write(json.dumps(result.model_dump(), default=str) + '\n')
                    elif format == 'json':
                        # For JSON, we'd need to handle array formatting
                        pass

        # Clear output file
        with open(output_file, 'w') as f:
            if format == 'json':
                f.write('[\n')
            elif format == 'jsonl':
                pass  # No header needed

        # Process in batches
        stats = self.process_in_batches(
            query_params,
            write_batch
        )

        # Close JSON array if needed
        if format == 'json':
            with open(output_file, 'a') as f:
                f.write('\n]')

        stats['output_file'] = output_file
        stats['format'] = format
        return stats