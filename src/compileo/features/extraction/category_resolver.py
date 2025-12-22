"""
Category Resolver Service

Handles category resolution and path mapping for taxonomy-based operations.
"""

from typing import Dict, List, Any, Optional
from src.compileo.features.extraction.context_models import HierarchicalCategory


class CategoryResolver:
    """
    Service for resolving category paths and IDs within taxonomy hierarchies.

    Provides utilities for:
    - Converting category IDs to category information
    - Resolving category paths to category objects
    - Checking category matches
    """

    def get_category_info_by_id(self, taxonomy: HierarchicalCategory, category_id: str) -> Optional[Dict[str, Any]]:
        """
        Get category information by ID from the taxonomy hierarchy.

        Args:
            taxonomy: Root taxonomy category
            category_id: Category ID to find

        Returns:
            Dictionary with category information or None if not found
        """
        for node in self._traverse_taxonomy(taxonomy):
            if hasattr(node, 'id') and node.id == category_id:
                return {
                    'id': node.id,
                    'name': node.name,
                    'description': getattr(node, 'description', ''),
                    'parent_path': getattr(node, 'parent_path', [])
                }
        return None

    def resolve_category_paths(self, taxonomy: HierarchicalCategory, category_paths: List[str]) -> List[HierarchicalCategory]:
        """
        Resolve category path strings to actual category objects.

        Args:
            taxonomy: Root taxonomy category
            category_paths: List of path strings to resolve

        Returns:
            List of resolved category objects
        """
        resolved_categories = []

        for path in category_paths:
            category = self._find_category_by_path(taxonomy, path)
            if category:
                resolved_categories.append(category)

        return resolved_categories

    def check_category_match(self, classifications: Dict[str, Any], target_categories: List[HierarchicalCategory]) -> List[str]:
        """
        Check if classifications match any of the target categories.

        Args:
            classifications: Classification results
            target_categories: Target categories to match against

        Returns:
            List of matched category paths
        """
        matched_paths = []

        for category in target_categories:
            if hasattr(category, 'id') and category.id in classifications:
                matched_paths.append(category.id)

        return matched_paths

    def _find_category_by_path(self, taxonomy: HierarchicalCategory, path: str) -> Optional[HierarchicalCategory]:
        """
        Find a category by its path string.

        Args:
            taxonomy: Root taxonomy category
            path: Path string (e.g., "cat_0_1_2" or "Technology.Software")

        Returns:
            Category object if found, None otherwise
        """
        # Handle index-based paths (e.g., "cat_0_1_2", "0_1_2")
        if path.startswith('cat_') or path.replace('_', '').isdigit():
            return self._find_by_index_path(taxonomy, path)

        # Handle name-based paths (e.g., "Technology.Software.Web Development")
        return self._find_by_name_path(taxonomy, path)

    def _find_by_index_path(self, taxonomy: HierarchicalCategory, path: str) -> Optional[HierarchicalCategory]:
        """
        Find category by index-based path.

        Args:
            taxonomy: Root taxonomy category
            path: Index path (e.g., "cat_0_1_2" or "0_1_2")

        Returns:
            Category object if found
        """
        # Clean the path
        if path.startswith('cat_'):
            path = path[4:]  # Remove 'cat_' prefix

        indices = [int(idx) for idx in path.split('_') if idx.isdigit()]

        current = taxonomy
        for idx in indices:
            if hasattr(current, 'children') and current.children and idx < len(current.children):
                current = current.children[idx]
            else:
                return None

        return current

    def _find_by_name_path(self, taxonomy: HierarchicalCategory, path: str) -> Optional[HierarchicalCategory]:
        """
        Find category by name-based path.

        Args:
            taxonomy: Root taxonomy category
            path: Name path (e.g., "Technology.Software.Web Development" or "RootName.Technology.Software")

        Returns:
            Category object if found
        """
        names = [name.strip() for name in path.split('.')]

        current = taxonomy

        # If the first name matches the root taxonomy name, skip it
        if names and hasattr(current, 'name') and current.name == names[0]:
            names = names[1:]

        for name in names:
            if hasattr(current, 'children') and current.children:
                found = False
                for child in current.children:
                    if hasattr(child, 'name') and child.name == name:
                        current = child
                        found = True
                        break
                if not found:
                    return None
            else:
                return None

        return current

    def _traverse_taxonomy(self, taxonomy: HierarchicalCategory):
        """
        Traverse the taxonomy tree and yield all nodes.

        Args:
            taxonomy: Root taxonomy node

        Yields:
            HierarchicalCategory: Each node in the taxonomy tree
        """
        yield taxonomy
        if hasattr(taxonomy, 'children') and taxonomy.children:
            for child in taxonomy.children:
                yield from self._traverse_taxonomy(child)