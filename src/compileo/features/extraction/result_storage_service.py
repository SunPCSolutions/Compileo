"""
Integrated service for extraction result storage, retrieval, and management.
Provides a unified API for all result storage operations.
"""

from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime

from .models import ExtractionResult, ExtractionResultMetadata
from .storage import (
    ExtractionResultStorageManager,
    TimeBasedChunkingStrategy,
    SizeBasedChunkingStrategy,
    CategoryBasedIndexingStrategy,
    FullTextIndexingStrategy,
    MemoryResultCache,
    TimeBasedRetentionPolicy,
    SizeBasedRetentionPolicy
)
from .result_search import ResultSearchEngine
from .result_cleanup import ResultCleanupService
from .result_metadata import ResultMetadataTracker


class ExtractionResultStorageService:
    """Unified service for extraction result storage operations."""

    def __init__(self, db_connection):
        self.db_connection = db_connection

        # Initialize components
        self.chunking_strategy = TimeBasedChunkingStrategy()
        self.indexing_strategy = CategoryBasedIndexingStrategy()
        self.cache = MemoryResultCache()
        self.retention_policy = TimeBasedRetentionPolicy(retention_days=30)

        # Initialize manager
        self.storage_manager = ExtractionResultStorageManager(
            chunking_strategy=self.chunking_strategy,
            indexing_strategy=self.indexing_strategy,
            retention_policy=self.retention_policy,
            cache=self.cache,
            db_connection=db_connection
        )

        # Initialize supporting services
        self.search_engine = ResultSearchEngine(db_connection)
        self.cleanup_service = ResultCleanupService(
            db_connection=db_connection,
            retention_policy=self.retention_policy
        )
        self.metadata_tracker = ResultMetadataTracker(db_connection)

        # Start background services
        self.storage_manager.start_background_cleanup()

    def store_results(self, results: List[ExtractionResult]) -> List[str]:
        """Store extraction results with chunking and indexing."""
        return self.storage_manager.store_results(results)

    def search_results(
        self,
        query: str = "",
        categories: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search extraction results with full-text and filters."""
        results, total_count = self.search_engine.full_text_search(
            query=query,
            categories=categories,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset
        )

        return {
            'results': results,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'has_more': offset + len(results) < total_count
        }

    def get_results_by_categories(
        self,
        categories: List[str],
        match_all: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get results filtered by categories."""
        results, total_count = self.search_engine.search_by_categories(
            categories=categories,
            match_all=match_all,
            limit=limit,
            offset=offset
        )

        return {
            'results': results,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'has_more': offset + len(results) < total_count
        }

    def get_results_by_chunk(
        self,
        chunk_id: str,
        min_confidence: Optional[float] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get results from a specific chunk."""
        results, total_count = self.search_engine.search_by_chunk(
            chunk_id=chunk_id,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset
        )

        return {
            'results': results,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'has_more': offset + len(results) < total_count
        }

    def get_chunk_metadata(self, chunk_id: str) -> Optional[ExtractionResultMetadata]:
        """Get metadata for a chunk."""
        return self.storage_manager.get_chunk_metadata(chunk_id)

    def get_comprehensive_stats(
        self,
        project_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive statistics about extraction results."""
        return self.metadata_tracker.get_comprehensive_stats(
            project_id=project_id,
            date_from=date_from,
            date_to=date_to
        )

    def get_category_stats(self, project_id: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics about categories."""
        return self.search_engine.get_category_stats(project_id=project_id)

    def get_chunk_stats(self, project_id: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics about chunks."""
        return self.search_engine.get_chunk_stats(project_id=project_id)

    def perform_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """Perform cleanup of old results."""
        return self.cleanup_service.perform_cleanup(force=force)

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup service statistics."""
        return self.cleanup_service.get_cleanup_stats()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage manager statistics."""
        return self.storage_manager.get_storage_stats()

    def get_result_size_estimate(
        self,
        project_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Estimate storage size of results."""
        return self.metadata_tracker.get_result_size_estimate(
            project_id=project_id,
            date_from=date_from,
            date_to=date_to
        )

    def configure_chunking_strategy(self, strategy_type: str, **kwargs) -> None:
        """Configure the chunking strategy."""
        if strategy_type == 'time_based':
            self.chunking_strategy = TimeBasedChunkingStrategy()
        elif strategy_type == 'size_based':
            chunk_size = kwargs.get('chunk_size', 100)
            self.chunking_strategy = SizeBasedChunkingStrategy()
            # Note: SizeBasedChunkingStrategy uses chunk_size in chunk_results method
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy_type}")

        # Update storage manager
        self.storage_manager.chunking_strategy = self.chunking_strategy

    def configure_indexing_strategy(self, strategy_type: str) -> None:
        """Configure the indexing strategy."""
        if strategy_type == 'category_based':
            self.indexing_strategy = CategoryBasedIndexingStrategy()
        elif strategy_type == 'full_text':
            self.indexing_strategy = FullTextIndexingStrategy()
        else:
            raise ValueError(f"Unknown indexing strategy: {strategy_type}")

        # Update storage manager
        self.storage_manager.indexing_strategy = self.indexing_strategy

    def configure_retention_policy(self, policy_type: str, **kwargs) -> None:
        """Configure the retention policy."""
        if policy_type == 'time_based':
            retention_days = kwargs.get('retention_days', 30)
            self.retention_policy = TimeBasedRetentionPolicy(retention_days=retention_days)
        elif policy_type == 'size_based':
            max_size_bytes = kwargs.get('max_size_bytes', 100 * 1024 * 1024)  # 100MB
            self.retention_policy = SizeBasedRetentionPolicy(max_size_bytes=max_size_bytes)
        else:
            raise ValueError(f"Unknown retention policy: {policy_type}")

        # Update services
        self.storage_manager.retention_policy = self.retention_policy
        self.cleanup_service.retention_policy = self.retention_policy

    def shutdown(self) -> None:
        """Shutdown the service and cleanup resources."""
        self.storage_manager.stop_background_cleanup()
        self.cleanup_service.stop_background_cleanup()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()