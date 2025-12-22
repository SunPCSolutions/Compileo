"""
Taxonomy extraction models and data structures.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ExtractionResult:
    """Result of taxonomy-based content extraction."""
    chunk_id: str
    chunk_text: str
    classifications: Dict[str, Any]
    confidence_score: float
    categories_matched: List[str]
    metadata: Dict[str, Any]


@dataclass
class ExtractionSummary:
    """Summary of extraction operation."""
    total_chunks: int
    processed_chunks: int
    filtered_chunks: int
    categories_used: List[str]
    confidence_threshold: float
    extraction_time: float
    results_file: Optional[str] = None