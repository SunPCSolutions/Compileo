"""
Response parser for taxonomy generation.
"""

import json
import re
from typing import Dict, Any, Optional, List


class TaxonomyResponseParser:
    """
    Parser for taxonomy generation responses from AI models.
    """

    @staticmethod
    def parse_taxonomy_response(response_text: str, category_limits: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Parse and validate Grok's taxonomy response.

        Args:
            response_text: Raw response from Grok
            category_limits: Optional limits for categories per level

        Returns:
            Validated taxonomy dictionary

        Raises:
            Exception: If parsing fails
        """
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")

            json_text = response_text[start_idx:end_idx]
            taxonomy = json.loads(json_text)

            # Validate structure
            TaxonomyResponseParser._validate_taxonomy_structure(taxonomy)

            # Enforce category limits if provided
            if category_limits:
                taxonomy = TaxonomyResponseParser._enforce_category_limits(taxonomy, category_limits)

            # Add hierarchical IDs to taxonomy
            taxonomy_with_ids = TaxonomyResponseParser._add_hierarchical_ids(taxonomy)

            return taxonomy_with_ids

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract any valid JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            raise Exception(f"Failed to parse taxonomy response: {e}")

    @staticmethod
    def parse_taxonomy_extension_response(response_text: str, existing_taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate Grok's taxonomy extension response.

        Args:
            response_text: Raw response from Grok
            existing_taxonomy: The original taxonomy for reference

        Returns:
            Extended taxonomy dictionary

        Raises:
            Exception: If parsing fails
        """
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")

            json_text = response_text[start_idx:end_idx]
            extended_taxonomy = json.loads(json_text)

            # Validate structure
            TaxonomyResponseParser._validate_taxonomy_structure(extended_taxonomy)

            return extended_taxonomy

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract any valid JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            raise Exception(f"Failed to parse taxonomy extension response: {e}")

    @staticmethod
    def _add_hierarchical_ids(taxonomy: Dict[str, Any], project_id: Any = 1, taxonomy_index: int = 1) -> Dict[str, Any]:
        """
        Add hierarchical IDs to taxonomy categories.

        ID format: {project_id}{taxonomy_index}{category_indices}
        Example: Project 1, Taxonomy 1, Category 0: "110"
                 Project 1, Taxonomy 1, Category 0, Subcategory 1: "1101"

        Args:
            taxonomy: Taxonomy dictionary
            project_id: Project identifier (can be string or int)
            taxonomy_index: Taxonomy index within project

        Returns:
            Taxonomy with IDs added
        """
        def add_ids_recursive(node: Dict[str, Any], current_path: str) -> Dict[str, Any]:
            # Add ID to current node
            node = node.copy()
            node['id'] = current_path

            # Process children
            if 'children' in node and isinstance(node['children'], list):
                updated_children = []
                for i, child in enumerate(node['children']):
                    child_path = f"{current_path}{i}"
                    updated_children.append(add_ids_recursive(child, child_path))
                node['children'] = updated_children

            return node

        # Start with root taxonomy
        root_id = f"{project_id}{taxonomy_index}"
        return add_ids_recursive(taxonomy, root_id)

    @staticmethod
    def _validate_taxonomy_structure(taxonomy: Dict[str, Any], max_depth: int = 5) -> None:
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

    @staticmethod
    def _enforce_category_limits(taxonomy: Dict[str, Any], category_limits: List[int]) -> Dict[str, Any]:
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
