"""
Taxonomy Module

This module provides comprehensive taxonomy management for the document processing pipeline,
sitting between the chunk module and extraction module. It handles taxonomy generation,
merging, and loading with full support for GUI workflows.

Main Components:
- TaxonomyProcessor: Main entry point for taxonomy operations
- TaxonomyGenerator: AI-powered taxonomy generation from chunks
- TaxonomyMerger: Intelligent merging of multiple taxonomies
- TaxonomyLoader: Taxonomy persistence and retrieval
"""

from .processor import TaxonomyProcessor
from .generator import TaxonomyGenerator
from .merger import TaxonomyMerger
from .loader import TaxonomyLoader

# New modular components
from .api_client import GrokAPIClient
from .prompt_builder import TaxonomyPromptBuilder
from .response_parser import TaxonomyResponseParser
from .analytics import TaxonomyAnalytics
from .content_processor import ContentProcessor

__all__ = [
    'TaxonomyProcessor',
    'TaxonomyGenerator',
    'TaxonomyMerger',
    'TaxonomyLoader',
    # New modular components
    'GrokAPIClient',
    'TaxonomyPromptBuilder',
    'TaxonomyResponseParser',
    'TaxonomyAnalytics',
    'ContentProcessor'
]