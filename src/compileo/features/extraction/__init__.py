# This file marks the 'extraction' directory as a Python package.

from .models import (
    ExtractionJob,
    ExtractionResult,
    ExtractionResultChunk,
    ExtractionResultMetadata,
    ExtractionJobStatus,
    serialize_extraction_result,
    deserialize_extraction_result,
    serialize_extraction_job,
    deserialize_extraction_job,
    serialize_result_chunk,
    deserialize_result_chunk,
)

from .storage import (
    ResultChunkingStrategy,
    TimeBasedChunkingStrategy,
    SizeBasedChunkingStrategy,
    ResultIndexingStrategy,
    CategoryBasedIndexingStrategy,
    FullTextIndexingStrategy,
    ResultCacheInterface,
    MemoryResultCache,
    ResultRetentionPolicy,
    TimeBasedRetentionPolicy,
    SizeBasedRetentionPolicy,
    ExtractionResultStorageManager,
)

from .result_search import ResultSearchEngine
from .result_cleanup import ResultCleanupService, TimeBasedCleanupScheduler
from .result_metadata import ResultMetadataTracker
from .result_storage_service import ExtractionResultStorageService

__all__ = [
    # Models
    'ExtractionJob',
    'ExtractionResult',
    'ExtractionResultChunk',
    'ExtractionResultMetadata',
    'ExtractionJobStatus',
    # Serialization
    'serialize_extraction_result',
    'deserialize_extraction_result',
    'serialize_extraction_job',
    'deserialize_extraction_job',
    'serialize_result_chunk',
    'deserialize_result_chunk',
    # Storage interfaces
    'ResultChunkingStrategy',
    'TimeBasedChunkingStrategy',
    'SizeBasedChunkingStrategy',
    'ResultIndexingStrategy',
    'CategoryBasedIndexingStrategy',
    'FullTextIndexingStrategy',
    'ResultCacheInterface',
    'MemoryResultCache',
    'ResultRetentionPolicy',
    'TimeBasedRetentionPolicy',
    'SizeBasedRetentionPolicy',
    'ExtractionResultStorageManager',
    # Search and analytics
    'ResultSearchEngine',
    # Cleanup and retention
    'ResultCleanupService',
    'TimeBasedCleanupScheduler',
    # Metadata tracking
    'ResultMetadataTracker',
    # Unified service
    'ExtractionResultStorageService',
]