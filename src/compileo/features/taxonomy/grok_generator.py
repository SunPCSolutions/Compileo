"""
AI-powered taxonomy generation using xAI Grok.
"""

import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from collections import Counter

from src.compileo.features.extraction.context_models import HierarchicalCategory
from .api_client import GrokAPIClient
from .prompt_builder import TaxonomyPromptBuilder
from .response_parser import TaxonomyResponseParser
from .analytics import TaxonomyAnalytics
from .content_processor import ContentProcessor


class GrokTaxonomyGenerator:
    """
    Generates hierarchical taxonomies from document chunks using xAI Grok.
    """

    def __init__(self, grok_api_key: str, model: Optional[str] = None):
        """
        Initialize the taxonomy generator.

        Args:
            grok_api_key: xAI Grok API key
            model: Grok model to use (optional, defaults to settings or "grok-4-fast-reasoning")
        """
        from src.compileo.core.settings import backend_settings
        if model is None:
            model = backend_settings.get_taxonomy_grok_model()
        self.api_client = GrokAPIClient(grok_api_key, model)
        self.model = model

    def generate_taxonomy(
        self,
        chunks: List[str],
        domain: str = "general",
        depth: Optional[int] = 3,
        batch_size: Optional[int] = None,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1,
        processing_mode: str = "fast"
    ) -> Dict[str, Any]:
        """
        Generate a hierarchical taxonomy from document chunks.

        Args:
            chunks: List of text chunks to analyze
            domain: Knowledge domain (medical, legal, technical, etc.)
            depth: Maximum hierarchy depth
            batch_size: Number of complete chunks to process in this batch (None = use all)
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories
            processing_mode: 'fast' (single pass) or 'complete' (iterative refinement)

        Returns:
            Extended taxonomy data with metadata and analytics
        """
        # Handle 'complete' mode using IterativeTaxonomyGenerator
        if processing_mode == "complete":
            from .iterative_generator import IterativeTaxonomyGenerator
            iterative_gen = IterativeTaxonomyGenerator(
                generation_func=self._generate_single_pass,
                extension_func=self.extend_taxonomy
            )
            return iterative_gen.generate(
                chunks=chunks,
                domain=domain,
                depth=depth,
                batch_size=batch_size or 10,
                category_limits=category_limits,
                specificity_level=specificity_level
            )

        return self._generate_single_pass(
            chunks=chunks,
            domain=domain,
            depth=depth,
            batch_size=batch_size,
            category_limits=category_limits,
            specificity_level=specificity_level
        )

    def _generate_single_pass(
        self,
        chunks: List[str],
        domain: str = "general",
        depth: Optional[int] = 3,
        batch_size: Optional[int] = None,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1
    ) -> Dict[str, Any]:
        """
        Generate taxonomy in a single pass (legacy/fast mode).
        """
        start_time = time.time()

        # Take batch of complete chunks if requested
        if batch_size and len(chunks) > batch_size:
            chunks = chunks[:batch_size]

        # Prepare content sample
        sample_content = ContentProcessor.prepare_content_sample(chunks)

        # Build prompt
        prompt = TaxonomyPromptBuilder.build_taxonomy_prompt(
            sample_content, domain, depth, category_limits, specificity_level
        )

        # Generate taxonomy using Grok
        taxonomy_data = self.api_client.call_for_taxonomy_generation(prompt)

        # Parse and validate the response
        taxonomy = TaxonomyResponseParser.parse_taxonomy_response(taxonomy_data, category_limits)

        # Generate analytics
        analytics = TaxonomyAnalytics.generate_analytics(chunks, taxonomy)

        # Create generation metadata
        generation_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "generator": f"grok-{self.model}",
            "domain": domain,
            "depth": depth,
            "specificity_level": specificity_level,
            "batch_size": len(chunks),
            "source_chunks": len(chunks),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "model_version": self.model,
            "confidence_score": TaxonomyAnalytics.calculate_overall_confidence(taxonomy)
        }

        return {
            "taxonomy": taxonomy,
            "generation_metadata": generation_metadata,
            "analytics": analytics
        }

    def extend_taxonomy(
        self,
        existing_taxonomy: Dict[str, Any],
        chunks: List[str],
        additional_depth: Optional[int] = 2,
        domain: str = "general",
        batch_size: Optional[int] = None,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1
    ) -> Dict[str, Any]:
        """
        Extend an existing taxonomy with additional depth levels using AI.

        Args:
            existing_taxonomy: The existing taxonomy to extend
            chunks: List of text chunks to analyze for extension
            additional_depth: Number of additional levels to add
            domain: Knowledge domain (medical, legal, technical, etc.)
            batch_size: Number of complete chunks to process in this batch (None = use all)
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories

        Returns:
            Extended taxonomy data with metadata and analytics
        """
        start_time = time.time()

        # Take batch of complete chunks if requested
        if batch_size and len(chunks) > batch_size:
            chunks = chunks[:batch_size]

        # Prepare content sample
        sample_content = ContentProcessor.prepare_content_sample(chunks)

        # Build extension prompt
        prompt = TaxonomyPromptBuilder.build_taxonomy_extension_prompt(
            existing_taxonomy, sample_content, additional_depth, domain, category_limits, specificity_level
        )

        # Generate extended taxonomy using Grok
        taxonomy_data = self.api_client.call_for_taxonomy_extension(prompt)

        # Parse and validate the response
        extended_taxonomy = TaxonomyResponseParser.parse_taxonomy_extension_response(taxonomy_data, existing_taxonomy)

        # Generate analytics
        analytics = TaxonomyAnalytics.generate_analytics(chunks, extended_taxonomy)

        # Create generation metadata
        generation_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "generator": f"grok-{self.model}",
            "domain": domain,
            "additional_depth": additional_depth,
            "specificity_level": specificity_level,
            "batch_size": len(chunks),
            "source_chunks": len(chunks),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "model_version": self.model,
            "confidence_score": TaxonomyAnalytics.calculate_overall_confidence(extended_taxonomy),
            "extension_type": "taxonomy_extension"
        }

        return {
            "taxonomy": extended_taxonomy,
            "generation_metadata": generation_metadata,
            "analytics": analytics
        }