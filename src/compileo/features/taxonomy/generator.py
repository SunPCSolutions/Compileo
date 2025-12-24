"""
AI-powered taxonomy generation using multiple AI providers.

This module provides functionality to automatically generate hierarchical taxonomies
from document chunks using various AI providers (Grok, Gemini, Ollama).
"""

from typing import List, Dict, Any, Optional, Union
from .grok_generator import GrokTaxonomyGenerator
from .gemini_generator import GeminiTaxonomyGenerator
from .ollama_generator import OllamaTaxonomyGenerator
from .openai_generator import OpenAITaxonomyGenerator


class TaxonomyGenerator:
    """
    Main taxonomy generator that delegates to specific AI provider generators.
    """

    def __init__(self, grok_api_key: str, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the taxonomy generator with available AI providers.

        Args:
            grok_api_key: xAI Grok API key
            gemini_api_key: Google Gemini API key (optional)
            openai_api_key: OpenAI API key (optional)
            model: Default model to use (optional)
        """
        self.generators: Dict[str, Union[GrokTaxonomyGenerator, GeminiTaxonomyGenerator, OllamaTaxonomyGenerator, OpenAITaxonomyGenerator]] = {
            'grok': GrokTaxonomyGenerator(grok_api_key, model or "grok-4-fast-reasoning"),
        }

        if gemini_api_key:
            self.generators['gemini'] = GeminiTaxonomyGenerator(gemini_api_key, model or "gemini-2.5-flash")

        if openai_api_key:
            self.generators['openai'] = OpenAITaxonomyGenerator(openai_api_key, model or "gpt-4o")
        else:
            # Still add OpenAI generator but it will fail at runtime if no API key
            self.generators['openai'] = OpenAITaxonomyGenerator("", model or "gpt-4o")

        # Ollama doesn't require API key
        self.generators['ollama'] = OllamaTaxonomyGenerator(model or "mistral:latest")

    def generate_taxonomy(
        self,
        chunks: List[str],
        domain: str = "general",
        depth: Optional[int] = 3,
        batch_size: Optional[int] = None,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1,
        provider: str = "grok",
        processing_mode: str = "fast"
    ) -> Dict[str, Any]:
        """
        Generate a hierarchical taxonomy from document chunks using specified AI provider.

        Args:
            chunks: List of text chunks to analyze
            domain: Knowledge domain (medical, legal, technical, etc.)
            depth: Maximum hierarchy depth
            batch_size: Number of complete chunks to process in this batch (None = use all)
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories
            provider: AI provider to use ('grok', 'gemini', 'ollama')
            processing_mode: 'fast' (single batch) or 'complete' (iterative refinement)

        Returns:
            Extended taxonomy data with metadata and analytics
        """
        if provider not in self.generators:
            raise ValueError(f"Unsupported provider: {provider}. Available: {list(self.generators.keys())}")

        generator = self.generators[provider]
        
        # Check if the generator supports processing_mode
        if hasattr(generator, 'generate_taxonomy') and 'processing_mode' in generator.generate_taxonomy.__code__.co_varnames:
            return generator.generate_taxonomy(
                chunks=chunks,
                domain=domain,
                depth=depth,
                batch_size=batch_size,
                category_limits=category_limits,
                specificity_level=specificity_level,
                processing_mode=processing_mode
            )
        else:
            # Fallback for generators that haven't been updated yet
            return generator.generate_taxonomy(
                chunks=chunks,
                domain=domain,
                depth=depth,
                batch_size=batch_size,
                category_limits=category_limits,
                specificity_level=specificity_level
            )

    def extend_taxonomy(
        self,
        existing_taxonomy: Dict[str, Any],
        chunks: List[str],
        additional_depth: Optional[int] = 2,
        domain: str = "general",
        batch_size: Optional[int] = None,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1,
        processing_mode: str = "fast",
        provider: str = "grok"
    ) -> Dict[str, Any]:
        """
        Extend an existing taxonomy with additional depth levels using specified AI provider.

        Args:
            existing_taxonomy: The existing taxonomy to extend
            chunks: List of text chunks to analyze for extension
            additional_depth: Number of additional levels to add
            domain: Knowledge domain (medical, legal, technical, etc.)
            batch_size: Number of complete chunks to process in this batch (None = use all)
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories
            provider: AI provider to use ('grok', 'gemini', 'ollama')

        Returns:
            Extended taxonomy data with metadata and analytics
        """
        if provider not in self.generators:
            raise ValueError(f"Unsupported provider: {provider}. Available: {list(self.generators.keys())}")

        generator = self.generators[provider]
        return generator.extend_taxonomy(
            existing_taxonomy=existing_taxonomy,
            chunks=chunks,
            additional_depth=additional_depth,
            domain=domain,
            batch_size=batch_size,
            category_limits=category_limits,
            specificity_level=specificity_level,
            processing_mode=processing_mode
        )
