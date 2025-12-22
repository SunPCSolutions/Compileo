"""
Category path resolution utilities for taxonomy extraction.
"""

from typing import List, Optional
from src.compileo.features.extraction.context_models import HierarchicalCategory
from src.compileo.features.extraction.exceptions import TaxonomyCategoryError
from src.compileo.features.extraction.error_logging import extraction_logger


class CategoryResolver:
    """
    Handles resolution of category path strings to actual category objects.
    """

    @staticmethod
    def resolve_category_paths(taxonomy: HierarchicalCategory, category_paths: List[str]) -> List[HierarchicalCategory]:
        """
        Resolve category path strings to actual category objects.
        Supports multiple path formats:
        - Index-based: "cat_0_1_2" or "0_1_2"
        - Name-based: "Technology.Software.Web Development"

        Args:
            taxonomy: Root taxonomy category
            category_paths: List of path strings.

        Returns:
            List of resolved category objects.

        Raises:
            TaxonomyCategoryError: If category path resolution fails for all paths.
        """
        resolved_categories = []
        failed_paths = []

        for path_str in category_paths:
            try:
                if not isinstance(path_str, str) or not path_str.strip():
                    raise ValueError("Invalid category path: must be a non-empty string")

                current = taxonomy
                # Check if path is index-based or name-based
                if (path_str.startswith('cat_') and all(part.isdigit() for part in path_str[4:].split('_'))) or \
                   (not path_str.startswith('cat_') and all(part.isdigit() for part in path_str.split('_'))):
                    # Index-based path (with or without "cat_" prefix)
                    if path_str.startswith('cat_'):
                        path_part = path_str[4:]
                    else:
                        path_part = path_str

                    indices = [int(x) for x in path_part.split('_') if x.strip()]
                    if not indices:
                        raise ValueError(f"Empty path indices in '{path_str}'")

                    # For index-based paths, traverse from the root using the indices
                    # The taxonomy root is the container, so we start traversal from its children
                    search_node = current.children if current.children else []

                    for i, idx in enumerate(indices):
                        if not search_node or idx < 0 or idx >= len(search_node):
                            raise IndexError(f"Category index {idx} out of range at level {i} in path '{path_str}'")
                        current = search_node[idx]
                        search_node = current.children if current.children else []
                else:
                    # Name-based path (dot-separated)
                    path_names = [name.strip() for name in path_str.split('.') if name.strip()]
                    if not path_names:
                        raise ValueError(f"Empty path names in '{path_str}'")

                    # Determine starting point: if path includes root name, start from root
                    if path_names[0] == current.name:
                        search_nodes = [current]
                    else:
                        search_nodes = current.children
                    for i, name in enumerate(path_names):
                        found = False
                        if search_nodes:
                            for node in search_nodes:
                                if node.name == name:
                                    current = node
                                    search_nodes = current.children
                                    found = True
                                    break
                        if not found:
                            raise ValueError(f"Category '{name}' not found at level {i} in path '{path_str}'")

                resolved_categories.append(current)

            except (ValueError, IndexError, AttributeError) as e:
                failed_paths.append(path_str)
                extraction_logger.log_warning(
                    f"Failed to resolve category path '{path_str}': {e}",
                    "resolve_category_paths",
                    context={"error_type": type(e).__name__, "path_str": path_str}
                )
                continue

        if failed_paths:
            extraction_logger.log_warning(
                f"Some category paths failed to resolve: {failed_paths}",
                "resolve_category_paths",
                context={"failed_paths": failed_paths, "successful_count": len(resolved_categories)}
            )

        return resolved_categories

    @staticmethod
    def check_category_match(classifications: dict, target_categories: List[HierarchicalCategory]) -> List[str]:
        """
        Check if classifications match any of the target categories.

        Args:
            classifications: Classification results
            target_categories: Target categories to match against

        Returns:
            List of matched category paths

        Raises:
            TaxonomyCategoryError: If category matching fails
        """
        if not isinstance(classifications, dict):
            raise TaxonomyCategoryError("Invalid classifications: must be a dictionary", "classifications")

        matched_paths = []

        for category in target_categories:
            try:
                category_name = category.name

                if not category_name or not isinstance(category_name, str):
                    extraction_logger.log_warning(
                        "Invalid category name",
                        "check_category_match",
                        context={"category_name": category_name, "category_type": type(category_name).__name__}
                    )
                    continue

                # Check if this category appears in classifications
                if category_name in classifications:
                    value = classifications[category_name]

                    # Convert value to string and check if it's meaningful
                    str_value = str(value).strip() if value is not None else ""
                    if str_value and str_value.lower() not in ('none', 'null', 'false', '0'):
                        try:
                            path_string = category.get_full_path_string()
                            matched_paths.append(path_string)
                        except Exception as e:
                            extraction_logger.log_error(
                                e,
                                "check_category_match",
                                context={"category_name": category_name, "error": "failed_to_get_path_string"}
                            )
                            continue

            except Exception as e:
                extraction_logger.log_error(
                    e,
                    "check_category_match",
                    context={"category_name": getattr(category, 'name', 'unknown')}
                )
                continue

        return matched_paths

    @staticmethod
    def category_belongs_to(classified_cat_name: str, category_id: str, taxonomy: HierarchicalCategory) -> bool:
        """
        Check if a category with the given ID belongs to a classified category.

        Args:
            classified_cat_name: Name of the classified category (e.g., "Diseases and Conditions")
            category_id: ID of the category to check (e.g., "11000")
            taxonomy: Root taxonomy category

        Returns:
            True if the category belongs to the classified category
        """
        def find_category_by_name(node: HierarchicalCategory, name: str) -> Optional[HierarchicalCategory]:
            """Find a category by name in the taxonomy."""
            if name.strip().lower() in node.name.strip().lower():
                return node
            if node.children:
                for child in node.children:
                    result = find_category_by_name(child, name)
                    if result:
                        return result
            return None

        def category_is_under_node(node: HierarchicalCategory, target_id: str) -> bool:
            """Check if a category ID is under this node."""
            if hasattr(node, 'id') and node.id == target_id:
                return True
            if node.children:
                for child in node.children:
                    if category_is_under_node(child, target_id):
                        return True
            return False

        # Find the classified category in the taxonomy
        classified_category = find_category_by_name(taxonomy, classified_cat_name)
        if not classified_category:
            extraction_logger.log_warning(
                f"Classified category '{classified_cat_name}' not found in taxonomy",
                "category_belongs_to",
                context={"classified_cat_name": classified_cat_name, "category_id": category_id}
            )
            return False

        # Check if the category ID is under this classified category
        result = category_is_under_node(classified_category, category_id)
        extraction_logger.log_operation_start(
            "category_belongs_to_check",
            context={
                "classified_cat_name": classified_cat_name,
                "category_id": category_id,
                "classified_category_found": classified_category.name if classified_category else None,
                "classified_category_id": getattr(classified_category, 'id', None),
                "result": result
            }
        )
        return result

    @staticmethod
    def get_category_info_by_id(taxonomy: HierarchicalCategory, search_id: str) -> Optional[dict]:
        """
        Get category information (name, description) by its ID.
        
        Args:
            taxonomy: Root taxonomy category
            search_id: ID to search for
            
        Returns:
            Dict with name and description, or None
        """
        def search_node(node: HierarchicalCategory) -> Optional[dict]:
            if hasattr(node, 'id') and node.id == search_id:
                return {
                    'name': node.name,
                    'description': getattr(node, 'description', '')
                }
            if node.children:
                for child in node.children:
                    result = search_node(child)
                    if result:
                        return result
            return None
        
        return search_node(taxonomy)

    def get_category_path_by_id(self, taxonomy: HierarchicalCategory, category_id: str) -> List[str]:
        """
        Get the hierarchical path of a category by its ID.

        Args:
            taxonomy: The root of the taxonomy tree.
            category_id: The ID of the category to find.

        Returns:
            A list of category names representing the path from the root to the category.
        """
        def find_path(node: HierarchicalCategory, target_id: str, current_path: List[str]) -> Optional[List[str]]:
            current_path.append(node.name)
            if hasattr(node, 'id') and node.id == target_id:
                return current_path
            
            if node.children:
                for child in node.children:
                    path = find_path(child, target_id, current_path.copy())
                    if path:
                        return path
            return None

        path = find_path(taxonomy, category_id, [])
        return path if path else []
        
    @staticmethod
    def find_category_by_id(taxonomy: HierarchicalCategory, search_id: str) -> Optional[str]:
        """
        Find a category name by its ID in the taxonomy.
        
        Args:
            taxonomy: Root taxonomy category
            search_id: ID to search for
            
        Returns:
            Category name or None
        """
        def search_node(node: HierarchicalCategory) -> Optional[str]:
            if hasattr(node, 'id') and node.id == search_id:
                return node.name
            if node.children:
                for child in node.children:
                    result = search_node(child)
                    if result:
                        return result
            return None
        
        return search_node(taxonomy)
    @staticmethod
    def find_category_by_name(taxonomy: HierarchicalCategory, search_name: str) -> Optional[str]:
        """
        Find a category in the taxonomy that matches the search name.
        
        Args:
            taxonomy: Root taxonomy category
            search_name: Name to search for
            
        Returns:
            Matching category name or None
        """
        def search_node(node: HierarchicalCategory) -> Optional[str]:
            if node.name and search_name.lower() in node.name.lower():
                return node.name
            if node.children:
                for child in node.children:
                    result = search_node(child)
                    if result:
                        return result
            return None
        
        return search_node(taxonomy)
