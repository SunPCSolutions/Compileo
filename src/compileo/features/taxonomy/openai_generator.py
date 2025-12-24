"""
AI-powered taxonomy generation using OpenAI (ChatGPT).
"""

import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from collections import Counter

from src.compileo.features.extraction.context_models import HierarchicalCategory


class OpenAITaxonomyGenerator:
    """
    Generates hierarchical taxonomies from document chunks using OpenAI.
    """

    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        Initialize the taxonomy generator.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.api_key = api_key
        self.model = model or "gpt-4o"
        self.client = None  # Will be initialized when needed

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise Exception("OpenAI package not installed. Please install with: pip install openai")
        return self.client

    def generate_taxonomy(
        self,
        chunks: List[str],
        domain: str = "general",
        depth: int = 3,
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
                batch_size=batch_size or 10,  # Default to 10 if None
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
        depth: int = 3,
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

        # Generate taxonomy using OpenAI
        taxonomy_data = self._call_openai_for_taxonomy(chunks, domain, depth, category_limits, specificity_level)

        # Parse and validate the response
        taxonomy = self._parse_taxonomy_response(taxonomy_data)

        # Enforce category limits if provided
        if category_limits:
            taxonomy = self._enforce_category_limits(taxonomy, category_limits)

        # Add hierarchical IDs to taxonomy
        taxonomy_with_ids = self._add_hierarchical_ids(taxonomy)

        # Generate analytics
        analytics = self._generate_analytics(chunks, taxonomy_with_ids)

        # Create generation metadata
        generation_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "generator": f"openai-{self.model}",
            "domain": domain,
            "depth": depth,
            "specificity_level": specificity_level,
            "batch_size": len(chunks),
            "source_chunks": len(chunks),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "model_version": self.model,
            "confidence_score": self._calculate_overall_confidence(taxonomy_with_ids)
        }

        return {
            "taxonomy": taxonomy_with_ids,
            "generation_metadata": generation_metadata,
            "analytics": analytics
        }

    def _call_openai_for_taxonomy(self, chunks: List[str], domain: str, depth: int, category_limits: Optional[List[int]] = None, specificity_level: int = 1) -> str:
        """
        Call OpenAI API to generate taxonomy from chunks.

        Args:
            chunks: Sampled text chunks
            domain: Knowledge domain
            depth: Maximum hierarchy depth

        Returns:
            Raw OpenAI response text
        """
        # Prepare sample content (limit to avoid token limits)
        sample_content = self._prepare_content_sample(chunks)

        # Create domain-specific prompt
        prompt = self._create_taxonomy_prompt(sample_content, domain, depth, category_limits, specificity_level)

        client = self._get_client()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert taxonomist specializing in knowledge organization and content classification."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=4000
                )
                content = response.choices[0].message.content

                if content:
                    return content
                else:
                    raise ValueError("Empty response from OpenAI API")

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to generate taxonomy after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        raise Exception("Unexpected error in taxonomy generation")

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

    def _create_taxonomy_prompt(self, content: str, domain: str, depth: int, category_limits: Optional[List[int]] = None, specificity_level: int = 1) -> str:
        """
        Create a domain-specific prompt for taxonomy generation.

        Args:
            content: Sampled content from chunks
            domain: Knowledge domain
            depth: Maximum hierarchy depth

        Returns:
            Complete prompt for OpenAI
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

        prompt = f"""You are an expert taxonomist specializing in {domain} knowledge organization. Your task is to analyze the provided content and create a hierarchical taxonomy that captures the key concepts, categories, and relationships within the {domain} domain.

**Domain Context:** {domain_context}

**Task Requirements:**
1. Analyze the content to identify main themes, concepts, and categories
2. Create a hierarchical taxonomy with maximum depth of {depth} levels
3. Ensure categories are specific to the {domain} domain
4. Use clear, descriptive names for categories
5. Include confidence scores (0.0-1.0) for each category based on content evidence
6. Assign unique hierarchical IDs to each category using this format: root is "11", first child is "110", second child is "111", grandchildren are "1100", "1101", etc.
7. Make categories mutually exclusive where possible
8. Focus on categories that would be useful for content classification and organization{specificity_instruction}{limits_instruction}

**Content to Analyze:**
{content}

**Output Format:**
Return a valid JSON object with this exact structure:
{{
  "name": "Root Category Name",
  "description": "Brief description of the root category",
  "id": "11",
  "children": [
    {{
      "name": "Level 1 Category",
      "description": "Description of this category",
      "id": "110",
      "confidence_threshold": 0.8,
      "children": [
        {{
          "name": "Level 2 Category",
          "description": "Description of subcategory",
          "id": "1100",
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
- Include id (string) and confidence_threshold (0.0-1.0) for each category
- Use hierarchical ID format: root="11", children="110","111", grandchildren="1100","1101", etc.
- Make category names specific and actionable"""

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
        Parse and validate OpenAI's taxonomy response.

        Args:
            response_text: Raw response from OpenAI

        Returns:
            Validated taxonomy dictionary
        """
        try:
            # Parse JSON directly from OpenAI response
            taxonomy = json.loads(response_text)

            # Validate structure
            self._validate_taxonomy_structure(taxonomy)

            return taxonomy

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract any valid JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            raise Exception(f"Failed to parse taxonomy response: {e}")

    def _validate_taxonomy_structure(self, taxonomy: Dict[str, Any], max_depth: int = 5) -> None:
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
            if current_depth > max_depth:
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
            return 0.5  # Default confidence

        # Weighted average favoring higher-level categories
        weights = []
        for i, score in enumerate(confidence_scores):
            # Give more weight to top-level categories
            level_weight = max(1.0, 2.0 - (i * 0.1))  # Decreasing weight
            weights.append(level_weight)

        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(confidence_scores, weights))

        return round(weighted_sum / total_weight, 3)

    def _add_hierarchical_ids(self, taxonomy: Dict[str, Any], project_id: int = 1, taxonomy_index: int = 1) -> Dict[str, Any]:
        """
        Add hierarchical integer IDs to taxonomy categories.

        ID format: {project_id}{taxonomy_index}{category_indices}
        Example: Project 1, Taxonomy 1, Category 0: "110"
                 Project 1, Taxonomy 1, Category 0, Subcategory 1: "1101"

        Args:
            taxonomy: Taxonomy dictionary
            project_id: Project identifier
            taxonomy_index: Taxonomy index within project

        Returns:
            Taxonomy with IDs added
        """
        import copy

        def add_ids_recursive(node: Dict[str, Any], current_path: str) -> Dict[str, Any]:
            # Deep copy the node to avoid reference issues
            new_node = copy.deepcopy(node)
            new_node['id'] = current_path

            # Process children recursively
            if 'children' in new_node and isinstance(new_node['children'], list):
                new_node['children'] = [
                    add_ids_recursive(child, f"{current_path}{i}")
                    for i, child in enumerate(new_node['children'])
                ]

            return new_node

        # Start with root taxonomy
        root_id = f"{project_id}{taxonomy_index}"
        return add_ids_recursive(taxonomy, root_id)

    def extend_taxonomy(
        self,
        existing_taxonomy: Dict[str, Any],
        chunks: List[str],
        additional_depth: int = 2,
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
            batch_size: Number of chunks to sample (None = use all)
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories

        Returns:
            Extended taxonomy data with metadata and analytics
        """
        start_time = time.time()

        # Sample chunks if requested
        if batch_size and len(chunks) > batch_size:
            chunks = chunks[:batch_size]

        # Generate extended taxonomy using OpenAI
        taxonomy_data = self._call_openai_for_taxonomy_extension(
            existing_taxonomy, chunks, additional_depth, domain, category_limits, specificity_level
        )

        # Parse and validate the response
        extended_taxonomy = self._parse_taxonomy_extension_response(taxonomy_data, existing_taxonomy)

        # Add hierarchical IDs to extended taxonomy if needed
        if 'id' not in extended_taxonomy:
            extended_taxonomy = self._add_hierarchical_ids(extended_taxonomy)

        # Generate analytics
        analytics = self._generate_analytics(chunks, extended_taxonomy)

        # Create generation metadata
        generation_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "generator": f"openai-{self.model}",
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

    def _call_openai_for_taxonomy_extension(
        self,
        existing_taxonomy: Dict[str, Any],
        chunks: List[str],
        additional_depth: int,
        domain: str,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1
    ) -> str:
        """
        Call OpenAI API to extend an existing taxonomy.

        Args:
            existing_taxonomy: The existing taxonomy to extend
            chunks: Sampled text chunks
            additional_depth: Number of additional levels to add
            domain: Knowledge domain
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories

        Returns:
            Raw OpenAI response text
        """
        # Prepare sample content (limit to avoid token limits)
        sample_content = self._prepare_content_sample(chunks)

        # Create domain-specific prompt for extension
        prompt = self._create_taxonomy_extension_prompt(
            existing_taxonomy, sample_content, additional_depth, domain, category_limits, specificity_level
        )

        client = self._get_client()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert taxonomist specializing in knowledge organization and content classification."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=4000
                )
                content = response.choices[0].message.content

                if content:
                    return content
                else:
                    raise ValueError("Empty response from OpenAI API")

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to extend taxonomy after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        raise Exception("Unexpected error in taxonomy extension")

    def _create_taxonomy_extension_prompt(
        self,
        existing_taxonomy: Dict[str, Any],
        content: str,
        additional_depth: int,
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
            Complete prompt for OpenAI
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

        prompt = f"""You are an expert taxonomist specializing in {domain} knowledge organization. Your task is to extend an existing taxonomy by adding {additional_depth} additional levels of subcategories based on the provided content.

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
- Preserve the existing taxonomy structure and add subcategories to appropriate nodes"""

        return prompt

    def _parse_taxonomy_extension_response(self, response_text: str, existing_taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate OpenAI's taxonomy extension response.

        Args:
            response_text: Raw response from OpenAI
            existing_taxonomy: The original taxonomy for reference

        Returns:
            Extended taxonomy dictionary
        """
        try:
            # Parse JSON directly from OpenAI response
            extended_taxonomy = json.loads(response_text)

            # Validate structure
            self._validate_taxonomy_structure(extended_taxonomy)

            return extended_taxonomy

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract any valid JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
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