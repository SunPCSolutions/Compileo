"""
AI-powered taxonomy generation using Ollama.
"""

import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from collections import Counter
from pydantic import BaseModel, Field

import ollama
from ...core.logging import get_logger

from src.compileo.features.extraction.context_models import HierarchicalCategory

logger = get_logger(__name__)


class TaxonomyCategory(BaseModel):
    """Pydantic model for taxonomy category with structured output."""
    name: str = Field(description="Category name")
    description: str = Field(description="Category description")
    confidence_threshold: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    children: List['TaxonomyCategory'] = Field(default_factory=list, description="Subcategories")


class TaxonomyOutput(BaseModel):
    """Pydantic model for complete taxonomy output."""
    name: str = Field(description="Root category name")
    description: str = Field(description="Root category description")
    children: List[TaxonomyCategory] = Field(description="Top-level categories")


class OllamaTaxonomyGenerator:
    """
    Generates hierarchical taxonomies from document chunks using Ollama.
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize the taxonomy generator.

        Args:
            model: Ollama model to use (if None, uses taxonomy-specific settings)
        """
        from ...core.settings import backend_settings
        # Use taxonomy-specific model if available, otherwise fall back to generation model
        self.model = model or backend_settings.get_setting("taxonomy_ollama_model") or backend_settings.get_generation_ollama_model()
        base_url = backend_settings.get_ollama_base_url()
        self.api_url = f"{base_url}/api/generate"
        self.backend_settings = backend_settings

    def generate_taxonomy(
        self,
        chunks: List[str],
        domain: str = "general",
        depth: Optional[int] = 3,
        batch_size: Optional[int] = None,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1,
        options: Optional[Dict[str, Any]] = None,
        processing_mode: str = "fast"
    ) -> Dict[str, Any]:
        """
        Generate a hierarchical taxonomy from document chunks.

        Args:
            chunks: List of text chunks to analyze
            domain: Knowledge domain (medical, legal, technical, etc.)
            depth: Maximum hierarchy depth
            batch_size: Number of complete chunks to process in this batch (None = use all)
            options: Optional Ollama API options to override defaults
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
            specificity_level=specificity_level,
            options=options
        )

    def _generate_single_pass(
        self,
        chunks: List[str],
        domain: str = "general",
        depth: Optional[int] = 3,
        batch_size: Optional[int] = None,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate taxonomy in a single pass (legacy/fast mode).
        """
        start_time = time.time()

        # Take batch of complete chunks if requested
        if batch_size and len(chunks) > batch_size:
            chunks = chunks[:batch_size]

        # Generate taxonomy using Ollama
        taxonomy_data = self._call_ollama_for_taxonomy(chunks, domain, depth, category_limits, specificity_level, options)

        # Parse and validate the response
        taxonomy = self._parse_taxonomy_response(taxonomy_data)

        # Enforce category limits if provided
        if category_limits:
            taxonomy = self._enforce_category_limits(taxonomy, category_limits)

        # Generate analytics
        analytics = self._generate_analytics(chunks, taxonomy)

        # Create generation metadata
        generation_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "generator": f"ollama-{self.model}",
            "domain": domain,
            "depth": depth,
            "specificity_level": specificity_level,
            "batch_size": len(chunks),
            "source_chunks": len(chunks),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "model_version": self.model,
            "confidence_score": self._calculate_overall_confidence(taxonomy)
        }

        return {
            "taxonomy": taxonomy,
            "generation_metadata": generation_metadata,
            "analytics": analytics
        }

    def _call_ollama_for_taxonomy(self, chunks: List[str], domain: str, depth: Optional[int], category_limits: Optional[List[int]] = None, specificity_level: int = 1, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Call Ollama API to generate taxonomy from chunks using chat API with structured output.
        Handles both thinking and non-thinking models.

        Args:
            chunks: Sampled text chunks
            domain: Knowledge domain
            depth: Maximum hierarchy depth
            options: Optional Ollama API options to override defaults

        Returns:
            JSON string of the taxonomy
        """
        # Prepare sample content (limit to avoid token limits)
        sample_content = self._prepare_content_sample(chunks)

        # Create domain-specific prompt
        prompt = self._create_taxonomy_prompt(sample_content, domain, depth, category_limits, specificity_level)

        # Get options from database via settings
        default_options = {
            "num_ctx": self.backend_settings.get_taxonomy_ollama_num_ctx(),
            "temperature": self.backend_settings.get_taxonomy_ollama_temperature(),
            "repeat_penalty": self.backend_settings.get_taxonomy_ollama_repeat_penalty(),
            "top_p": self.backend_settings.get_taxonomy_ollama_top_p(),
            "top_k": self.backend_settings.get_taxonomy_ollama_top_k(),
            "num_predict": self.backend_settings.get_taxonomy_ollama_num_predict()
        }
        
        seed = self.backend_settings.get_taxonomy_ollama_seed()
        if seed is not None:
            default_options["seed"] = seed

        # Merge provided options with defaults
        merged_options = {**default_options, **(options or {})}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Ollama chat API call attempt {attempt + 1}/{max_retries}")
                logger.debug(f"Model: {self.model}")
                logger.debug(f"Prompt length: {len(prompt)} characters")

                # Try to get content using adaptive approach (handles both thinking and regular models)
                try:
                    content = self._call_ollama_adaptive(prompt, merged_options)
                    logger.debug(f"Ollama response content length: {len(content)}")

                    if content:
                        logger.debug("Ollama chat API call successful")
                        return content
                    else:
                        logger.debug("Empty response from Ollama chat API")
                        raise ValueError("Empty response from Ollama chat API")
                except Exception as ollama_error:
                    logger.error(f"Ollama adaptive call failed: {ollama_error}")
                    raise Exception(f"Ollama taxonomy generation failed: {str(ollama_error)}")

            except Exception as e:
                logger.error(f"Ollama chat API call failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All Ollama attempts failed, raising exception")
                    raise Exception(f"Failed to generate taxonomy after {max_retries} attempts: {e}")
                logger.debug(f"Retrying Ollama call in {2 ** attempt} seconds")
                time.sleep(2 ** attempt)  # Exponential backoff

        raise Exception("Unexpected error in taxonomy generation")

    def _call_ollama_adaptive(self, prompt: str, options: Dict[str, Any]) -> str:
        """
        Adaptively call Ollama API, handling both thinking and regular models dynamically.

        Args:
            prompt: The prompt to send
            options: Ollama options

        Returns:
            Final content from the model
        """
        logger.debug(f"Trying adaptive Ollama call with model {self.model} - attempting thinking mode first")

        try:
            # First try thinking mode (works for both thinking and regular models)
            logger.debug("Attempting thinking mode...")
            content = self._call_with_thinking_mode(prompt, options)
            if content and self._is_valid_json(content):
                logger.debug("Thinking mode successful")
                return content
            else:
                logger.debug(f"Thinking mode returned invalid content: {content[:200]}...")
        except Exception as e:
            logger.debug(f"Thinking mode failed: {e}")

        logger.debug("Falling back to regular mode")
        try:
            # Fall back to regular mode
            logger.debug("Attempting regular mode...")
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format=TaxonomyOutput.model_json_schema(),
                options=options,
                stream=False
            )
            content = response.get('message', {}).get('content', '')
            logger.debug(f"Regular mode response content: {content[:200]}...")
            if content and self._is_valid_json(content):
                logger.debug("Regular mode successful")
                return content
            else:
                logger.debug(f"Regular mode returned invalid content: {content[:200]}...")
        except Exception as e:
            logger.debug(f"Regular mode also failed: {e}")

        raise Exception("Both thinking and regular modes failed to produce valid content")

    def _call_with_thinking_mode(self, prompt: str, options: Dict[str, Any]) -> str:
        """
        Call Ollama with thinking mode enabled, handling both thinking and regular models.

        Args:
            prompt: The prompt to send
            options: Ollama options

        Returns:
            Final content (thinking tokens filtered out)
        """
        logger.debug(f"Calling thinking mode with model {self.model}")
        try:
            stream = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format=TaxonomyOutput.model_json_schema(),
                options=options,
                think=True,  # Enable thinking - works for both types of models
                stream=True
            )

            content = ''
            in_thinking = False
            chunk_count = 0

            for chunk in stream:
                chunk_count += 1
                if chunk.message.thinking and not in_thinking:
                    in_thinking = True
                    logger.debug(f"Thinking phase detected at chunk {chunk_count}")

                if chunk.message.thinking:
                    # Skip thinking tokens
                    continue
                elif chunk.message.content:
                    if in_thinking:
                        in_thinking = False
                        logger.debug(f"Content phase started at chunk {chunk_count}")
                    content += chunk.message.content

            logger.debug(f"Thinking mode completed with {chunk_count} chunks, content length: {len(content)}")
            return content

        except Exception as e:
            logger.debug(f"Thinking mode stream failed: {e}")
            raise

    def _is_valid_json(self, content: str) -> bool:
        """
        Check if the content is valid JSON.

        Args:
            content: Content to validate

        Returns:
            True if valid JSON
        """
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


    def _prepare_content_sample(self, chunks: List[str], max_chars: int = 8000) -> str:
        """
        Prepare content from complete chunks in sequential order.
        Takes all available chunks up to a reasonable limit for token constraints.

        Args:
            chunks: All available chunks (already batched at generator level)
            max_chars: Maximum characters to include (may be exceeded for complete chunks)

        Returns:
            Concatenated content from complete chunks in order
        """
        if not chunks:
            return ""

        # Take complete chunks in sequential order - no sorting or prioritization
        # Use up to 10 complete chunks to ensure adequate content coverage
        # (batching is handled at the generator level)
        sample_parts = []
        total_chars = 0

        # Take chunks in the order they appear (up to 10 chunks for reasonable token limits)
        for chunk in chunks[:10]:
            sample_parts.append(chunk)
            total_chars += len(chunk)

        return "\n\n---\n\n".join(sample_parts)

    def _create_taxonomy_prompt(self, content: str, domain: str, depth: Optional[int], category_limits: Optional[List[int]] = None, specificity_level: int = 1) -> str:
        """
        Create a domain-specific prompt for taxonomy generation.

        Args:
            content: Sampled content from chunks
            domain: Knowledge domain
            depth: Maximum hierarchy depth

        Returns:
            Complete prompt for Ollama
        """
        domain_context = self._get_domain_context(domain)

        # Add specificity level guidance
        specificity_instruction = self._get_specificity_instruction(specificity_level)

        # Add category limits to the prompt if provided
        limits_instruction = ""
        if category_limits:
            limits_instruction = f"\n**Category Limits:**\n"
            for i, limit in enumerate(category_limits, 1):
                limits_instruction += f"- Level {i}: Maximum {limit} categories\n"
            limits_instruction += "\n"

        prompt = f"""[INST] You are an expert taxonomist specializing in {domain} knowledge organization. Your task is to analyze the provided content and create a hierarchical taxonomy that captures the key concepts, categories, and relationships within the {domain} domain.

**Domain Context:** {domain_context}

**Task Requirements:**
1. Analyze the content to identify main themes, concepts, and categories
2. Create a hierarchical taxonomy with maximum depth of {depth} levels
3. Ensure categories are specific to the {domain} domain
4. Use clear, descriptive names for categories
5. Include confidence scores (0.0-1.0) for each category based on content evidence
6. Make categories mutually exclusive where possible
7. Focus on categories that would be useful for content classification and organization{specificity_instruction}{limits_instruction}

**Content to Analyze:**
{content}

**Output Format:**
Return a valid JSON object with this exact structure:
{{
  "name": "Root Category Name",
  "description": "Brief description of the root category",
  "children": [
    {{
      "name": "Level 1 Category",
      "description": "Description of this category",
      "confidence_threshold": 0.8,
      "children": [
        {{
          "name": "Level 2 Category",
          "description": "Description of subcategory",
          "confidence_threshold": 0.7,
          "children": [...]
        }}
      ]
    }}
  ]
}}

**Important:**
- Return ONLY the JSON object, no other text
- Ensure the JSON is valid and properly formatted
- Maximum depth: {depth} levels
- Include confidence_threshold for each category (0.0-1.0)
- Make category names specific and actionable [/INST]"""

        return prompt

    def _get_domain_context(self, domain: str) -> str:
        """
        Get domain-specific context for taxonomy generation.

        Args:
            domain: Knowledge domain

        Returns:
            Domain-specific guidance text
        """
        contexts = {
            "medical": "Focus on clinical conditions, symptoms, treatments, diagnostics, anatomy, and medical specialties. Consider diseases, medications, procedures, and patient care aspects.",
            "legal": "Focus on legal domains, case types, jurisdictions, legal procedures, contracts, regulations, and legal concepts. Consider civil law, criminal law, corporate law, etc.",
            "technical": "Focus on technical domains, software development, hardware, processes, methodologies, and technical concepts. Consider programming, systems, infrastructure, etc.",
            "business": "Focus on business domains, industries, departments, processes, strategies, and business concepts. Consider finance, marketing, operations, management, etc.",
            "scientific": "Focus on scientific domains, research areas, methodologies, theories, and scientific concepts. Consider physics, chemistry, biology, mathematics, etc.",
            "general": "Create a general-purpose taxonomy suitable for most content types. Focus on universal categories like people, places, events, concepts, etc."
        }

        return contexts.get(domain.lower(), contexts["general"])

    def _get_specificity_instruction(self, specificity_level: int) -> str:
        """
        Get specificity level instructions for taxonomy generation.

        Args:
            specificity_level: Level from 1-5, where higher numbers mean more specific categories

        Returns:
            Specificity instruction text for the prompt
        """
        specificity_instructions = {
            1: "**Specificity Level 1 (Most General):** Create very broad, high-level categories that capture the most general themes. Use overarching concepts that would apply to many different contexts. Avoid overly specific or niche categories.",
            2: "**Specificity Level 2 (1 level higher than Level 1):** Create moderately broad categories that are 1 specificity level higher than Level 1. Balance generality with some specificity while remaining more general than Level 3. Include common sub-themes but avoid highly specialized terms.",
            3: "**Specificity Level 3 (1 level higher than Level 2):** Create categories with a good balance of breadth and specificity that are 1 specificity level higher than Level 2. Include both general concepts and moderately specific subcategories as appropriate for the content, but remain more general than Level 4.",
            4: "**Specificity Level 4 (1 level higher than Level 3):** Create more detailed and specific categories that are 1 specificity level higher than Level 3. Focus on particular aspects, subtypes, and specific implementations rather than broad generalizations, but remain more general than Level 5.",
            5: "**Specificity Level 5 (1 level higher than Level 4):** Create very detailed, granular categories that are 1 specificity level higher than Level 4. Use highly specific terms, particular methodologies, and specialized concepts. Include fine distinctions and specific use cases."
        }

        instruction = specificity_instructions.get(specificity_level, specificity_instructions[3])
        return f"\n{instruction}\n\n"

    def _parse_taxonomy_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate Ollama's taxonomy response from structured output.

        Args:
            response_text: JSON response from Ollama chat API

        Returns:
            Validated taxonomy dictionary
        """
        try:
            # The structured output should be valid JSON directly
            taxonomy = json.loads(response_text)

            # Validate structure
            self._validate_taxonomy_structure(taxonomy)

            return taxonomy

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract JSON if it's wrapped in other text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    taxonomy = json.loads(json_match.group())
                    self._validate_taxonomy_structure(taxonomy)
                    return taxonomy
                except (json.JSONDecodeError, ValueError):
                    pass

            raise Exception(f"Failed to parse taxonomy response: {e}")

    def _validate_taxonomy_structure(self, taxonomy: Dict[str, Any], max_depth: Optional[int] = 5) -> None:
        """
        Validate the structure of generated taxonomy.

        Args:
            taxonomy: Taxonomy dictionary to validate
            max_depth: Maximum allowed depth

        Raises:
            ValueError: If taxonomy structure is invalid
        """
        if not isinstance(taxonomy, dict):
            raise ValueError("Taxonomy must be a dictionary")

        required_keys = ["name", "description", "children"]
        for key in required_keys:
            if key not in taxonomy:
                raise ValueError(f"Missing required key: {key}")

        if not isinstance(taxonomy["children"], list):
            raise ValueError("Children must be a list")

        # Validate depth
        def check_depth(node: Dict[str, Any], current_depth: int = 0):
            if max_depth and current_depth > max_depth:
                raise ValueError(f"Taxonomy exceeds maximum depth of {max_depth}")
            for child in node.get("children", []):
                check_depth(child, current_depth + 1)

        check_depth(taxonomy)

    def _generate_analytics(self, chunks: List[str], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate analytics about the taxonomy and content.

        Args:
            chunks: Original chunks used for generation
            taxonomy: Generated taxonomy

        Returns:
            Analytics dictionary
        """
        # Count categories by level
        depth_counts = {}
        confidence_scores = []

        def analyze_node(node: Dict[str, Any], depth: int = 0):
            if depth not in depth_counts:
                depth_counts[depth] = 0
            depth_counts[depth] += 1

            if "confidence_threshold" in node:
                confidence_scores.append(node["confidence_threshold"])

            for child in node.get("children", []):
                analyze_node(child, depth + 1)

        analyze_node(taxonomy)

        # Calculate content statistics
        total_chars = sum(len(chunk) for chunk in chunks)
        total_words = sum(len(chunk.split()) for chunk in chunks)

        return {
            "category_distribution": {
                f"level_{depth}": count
                for depth, count in depth_counts.items()
            },
            "depth_analysis": {
                "max_depth": max(depth_counts.keys()) if depth_counts else 0,
                "total_categories": sum(depth_counts.values()),
                "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 3) if confidence_scores else 0
            },
            "content_coverage": {
                "chunks_analyzed": len(chunks),
                "total_characters": total_chars,
                "total_words": total_words,
                "avg_chunk_length": round(total_chars / len(chunks), 1) if chunks else 0
            }
        }

    def _calculate_overall_confidence(self, taxonomy: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for the taxonomy.

        Args:
            taxonomy: Generated taxonomy

        Returns:
            Overall confidence score (0.0-1.0)
        """
        confidence_scores = []

        def collect_confidence(node: Dict[str, Any]):
            if "confidence_threshold" in node:
                confidence_scores.append(node["confidence_threshold"])
            for child in node.get("children", []):
                collect_confidence(child)

        collect_confidence(taxonomy)

        if not confidence_scores:
            return 0.7  # Default confidence when no scores provided

        # Filter out invalid scores (0.0 or negative)
        valid_scores = [score for score in confidence_scores if score > 0.0]

        if not valid_scores:
            return 0.7  # Default confidence when all scores are 0.0 or invalid

        # Weighted average favoring higher-level categories
        weights = []
        for i, score in enumerate(valid_scores):
            # Give more weight to top-level categories
            level_weight = max(1.0, 2.0 - (i * 0.1))  # Decreasing weight
            weights.append(level_weight)

        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(valid_scores, weights))

        result = round(weighted_sum / total_weight, 3)
        # Ensure result is never 0.0
        return max(result, 0.1)

    def extend_taxonomy(
        self,
        existing_taxonomy: Dict[str, Any],
        chunks: List[str],
        additional_depth: Optional[int] = 2,
        domain: str = "general",
        batch_size: Optional[int] = None,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extend an existing taxonomy with additional depth levels using AI.

        Args:
            existing_taxonomy: The existing taxonomy to extend
            chunks: List of text chunks to analyze for extension
            additional_depth: Number of additional levels to add
            domain: Knowledge domain (medical, legal, technical, etc.)
            batch_size: Number of chunks to sample (None = use all)
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories
            options: Optional Ollama API options to override defaults
            **kwargs: Additional arguments for compatibility (e.g. sample_size, processing_mode)

        Returns:
            Extended taxonomy data with metadata and analytics
        """
        start_time = time.time()

        # Support both batch_size and sample_size for compatibility
        limit = batch_size or kwargs.get('sample_size')

        # Sample chunks if requested
        if limit and len(chunks) > limit:
            from src.compileo.features.extraction.chunk_sampler import ChunkSampler
            chunks = ChunkSampler.sample_chunks(chunks, limit)

        # Generate extended taxonomy using Ollama
        taxonomy_data = self._call_ollama_for_taxonomy_extension(
            existing_taxonomy, chunks, additional_depth, domain, category_limits, specificity_level, options
        )

        # Parse and validate the response
        extended_taxonomy = self._parse_taxonomy_extension_response(taxonomy_data, existing_taxonomy)

        # Generate analytics
        analytics = self._generate_analytics(chunks, extended_taxonomy)

        # Create generation metadata
        generation_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "generator": f"ollama-{self.model}",
            "domain": domain,
            "additional_depth": additional_depth,
            "specificity_level": specificity_level,
            "batch_size": len(chunks),
            "source_chunks": len(chunks),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "model_version": self.model,
            "confidence_score": self._calculate_overall_confidence(extended_taxonomy),
            "extension_type": "taxonomy_extension"
        }

        return {
            "taxonomy": extended_taxonomy,
            "generation_metadata": generation_metadata,
            "analytics": analytics
        }

    def _call_ollama_for_taxonomy_extension(
        self,
        existing_taxonomy: Dict[str, Any],
        chunks: List[str],
        additional_depth: Optional[int],
        domain: str,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Call Ollama API to extend an existing taxonomy using chat API with structured output.
        Handles both thinking and non-thinking models.

        Args:
            existing_taxonomy: The existing taxonomy to extend
            chunks: Sampled text chunks
            additional_depth: Number of additional levels to add
            domain: Knowledge domain
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories
            options: Optional Ollama API options to override defaults

        Returns:
            JSON string of the extended taxonomy
        """
        # Prepare sample content (limit to avoid token limits)
        sample_content = self._prepare_content_sample(chunks)

        # Create domain-specific prompt for extension
        prompt = self._create_taxonomy_extension_prompt(
            existing_taxonomy, sample_content, additional_depth, domain, category_limits, specificity_level
        )

        # Get options from database via settings
        default_options = {
            "num_ctx": self.backend_settings.get_taxonomy_ollama_num_ctx(),
            "temperature": self.backend_settings.get_taxonomy_ollama_temperature(),
            "repeat_penalty": self.backend_settings.get_taxonomy_ollama_repeat_penalty(),
            "top_p": self.backend_settings.get_taxonomy_ollama_top_p(),
            "top_k": self.backend_settings.get_taxonomy_ollama_top_k(),
            "num_predict": self.backend_settings.get_taxonomy_ollama_num_predict()
        }
        
        seed = self.backend_settings.get_taxonomy_ollama_seed()
        if seed is not None:
            default_options["seed"] = seed

        # Merge provided options with defaults
        merged_options = {**default_options, **(options or {})}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Ollama extension chat API call attempt {attempt + 1}/{max_retries}")
                logger.debug(f"Model: {self.model}")

                # Try to get content using adaptive approach
                content = self._call_ollama_adaptive(prompt, merged_options)

                if content:
                    logger.debug("Ollama extension chat API call successful")
                    return content
                else:
                    raise ValueError("Empty response from Ollama extension chat API")

            except Exception as e:
                logger.debug(f"Ollama extension chat API call failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to extend taxonomy after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        raise Exception("Unexpected error in taxonomy extension")

    def _create_taxonomy_extension_prompt(
        self,
        existing_taxonomy: Dict[str, Any],
        content: str,
        additional_depth: Optional[int],
        domain: str,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1
    ) -> str:
        """
        Create a prompt for extending an existing taxonomy.

        Args:
            existing_taxonomy: The existing taxonomy to extend
            content: Sampled content from chunks
            additional_depth: Number of additional levels to add
            domain: Knowledge domain
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories

        Returns:
            Complete prompt for Ollama
        """
        domain_context = self._get_domain_context(domain)
        specificity_instruction = self._get_specificity_instruction(specificity_level)

        # Add category limits to the prompt if provided
        limits_instruction = ""
        if category_limits:
            limits_instruction = f"\n**Category Limits per New Level:**\n"
            for i, limit in enumerate(category_limits, 1):
                limits_instruction += f"- New Level {i}: Maximum {limit} categories\n"
            limits_instruction += "\n"

        # Convert existing taxonomy to readable format
        existing_taxonomy_str = json.dumps(existing_taxonomy, indent=2)

        prompt = f"""[INST] You are an expert taxonomist specializing in {domain} knowledge organization. Your task is to extend an existing taxonomy by adding {additional_depth} additional levels of subcategories based on the provided content.

**Domain Context:** {domain_context}

**Existing Taxonomy to Extend:**
{existing_taxonomy_str}

**Task Requirements:**
1. Analyze the existing taxonomy structure and identify where subcategories can be added
2. Add {additional_depth} additional levels of subcategories to the existing taxonomy
3. Ensure new categories are specific to the {domain} domain and consistent with existing categories
4. Use clear, descriptive names for new categories
5. Include confidence scores (0.0-1.0) for each new category based on content evidence
6. Make new categories mutually exclusive where possible
7. Focus on categories that would be useful for content classification and organization{specificity_instruction}{limits_instruction}

**Content to Analyze for Extension:**
{content}

**Output Format:**
Return a valid JSON object representing the extended taxonomy with this exact structure:
{{
  "name": "Root Category Name",
  "description": "Brief description of the root category",
  "children": [
    {{
      "name": "Level 1 Category",
      "description": "Description of this category",
      "confidence_threshold": 0.8,
      "children": [
        {{
          "name": "Level 2 Category",
          "description": "Description of subcategory",
          "confidence_threshold": 0.7,
          "children": [...]
        }}
      ]
    }}
  ]
}}

**Important:**
- Return ONLY the JSON object, no other text
- Ensure the JSON is valid and properly formatted
- Add exactly {additional_depth} additional levels to the existing taxonomy
- Include confidence_threshold for each new category (0.0-1.0)
- Make category names specific and actionable
- Preserve the existing taxonomy structure and add subcategories to appropriate nodes [/INST]"""

        return prompt

    def _parse_taxonomy_extension_response(self, response_text: str, existing_taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate Ollama's taxonomy extension response from structured output.

        Args:
            response_text: JSON response from Ollama chat API
            existing_taxonomy: The original taxonomy for reference

        Returns:
            Extended taxonomy dictionary
        """
        try:
            # The structured output should be valid JSON directly
            extended_taxonomy = json.loads(response_text)

            # Validate structure
            self._validate_taxonomy_structure(extended_taxonomy)

            return extended_taxonomy

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract JSON if it's wrapped in other text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    extended_taxonomy = json.loads(json_match.group())
                    self._validate_taxonomy_structure(extended_taxonomy)
                    return extended_taxonomy
                except (json.JSONDecodeError, ValueError):
                    pass

            raise Exception(f"Failed to parse taxonomy extension response: {e}")

    def _enforce_category_limits(self, taxonomy: Dict[str, Any], category_limits: List[int]) -> Dict[str, Any]:
        """
        Enforce category limits by trimming excess categories at each level.
        Limits are maximums - AI can choose fewer categories if appropriate for content.

        Args:
            taxonomy: Taxonomy dictionary
            category_limits: List of maximum limits per level (e.g., [5, 3, 2])

        Returns:
            Taxonomy with limits enforced
        """
        def trim_children(node: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
            # Make a copy to avoid modifying the original
            node = node.copy()

            if 'children' in node and isinstance(node['children'], list):
                # Get the limit for this depth level
                if depth < len(category_limits):
                    limit = category_limits[depth]
                    # Only trim if children exceed the limit
                    if len(node['children']) > limit:
                        node['children'] = node['children'][:limit]

                # Recursively trim children of children
                node['children'] = [
                    trim_children(child, depth + 1)
                    for child in node['children']
                ]

            return node

        return trim_children(taxonomy)