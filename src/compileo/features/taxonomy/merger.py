"""
Taxonomy Merger

This module provides intelligent merging of taxonomies with source tracking
and conflict resolution for combining user-provided and AI-generated taxonomies.
"""

from typing import Dict, List, Any
from datetime import datetime, timezone
from src.compileo.features.extraction.context_models import HierarchicalCategory


class TaxonomyMerger:
    """Service for intelligently merging taxonomies with source tracking."""

    @staticmethod
    def merge_taxonomies(primary: HierarchicalCategory, secondary: HierarchicalCategory,
                        primary_source: str = "manual", secondary_source: str = "auto") -> HierarchicalCategory:
        """
        Merge two taxonomies with intelligent conflict resolution and source tracking.

        Args:
            primary: Primary taxonomy (usually user-provided)
            secondary: Secondary taxonomy (usually AI-generated)
            primary_source: Source label for primary taxonomy
            secondary_source: Source label for secondary taxonomy

        Returns:
            Merged taxonomy with source tracking
        """
        current_time = datetime.now(timezone.utc).isoformat() + "Z"

        # Create merged root
        merged_root = HierarchicalCategory(
            name="Merged Taxonomy",
            description=f"Combined from {primary_source} and {secondary_source} sources",
            source="merged",
            source_timestamp=current_time,
            merged_from=[primary.name, secondary.name],
            confidence_threshold=min(primary.confidence_threshold, secondary.confidence_threshold)
        )

        # Collect all categories with their full paths
        primary_categories = TaxonomyMerger._flatten_taxonomy(primary, primary_source)
        secondary_categories = TaxonomyMerger._flatten_taxonomy(secondary, secondary_source)

        # Create a map for quick lookup
        category_map = {}

        # Add primary categories first (preserve user preferences)
        for path, category in primary_categories.items():
            category_map[path] = category

        # Add secondary categories, merging where appropriate
        for path, category in secondary_categories.items():
            if path in category_map:
                # Category exists, enrich with secondary information
                existing = category_map[path]
                TaxonomyMerger._enrich_category(existing, category, secondary_source)
            else:
                # New category, add it
                category_map[path] = category

        # Rebuild hierarchy from flat map
        TaxonomyMerger._rebuild_hierarchy(merged_root, category_map)

        return merged_root

    @staticmethod
    def merge_raw_taxonomies(taxonomy1: Dict[str, Any], taxonomy2: Dict[str, Any],
                             source1: str = "base", source2: str = "refinement") -> Dict[str, Any]:
        """
        Merge two raw JSON taxonomy dictionaries.

        This is a helper method for working with raw API responses or JSON data
        without converting to HierarchicalCategory objects first.

        Args:
            taxonomy1: First taxonomy (base)
            taxonomy2: Second taxonomy (refinement/extension)
            source1: Source label for first taxonomy
            source2: Source label for second taxonomy

        Returns:
            Merged taxonomy dictionary
        """
        # Convert dictionaries to HierarchicalCategory objects
        root1 = TaxonomyMerger._dict_to_category(taxonomy1)
        root2 = TaxonomyMerger._dict_to_category(taxonomy2)

        # Use the existing merge logic
        merged_root = TaxonomyMerger.merge_taxonomies(root1, root2, source1, source2)

        # Convert back to dictionary
        return TaxonomyMerger._category_to_dict(merged_root)

    @staticmethod
    def _dict_to_category(data: Dict[str, Any]) -> HierarchicalCategory:
        """Convert a dictionary to a HierarchicalCategory object."""
        category = HierarchicalCategory(
            name=data.get('name', 'Unknown'),
            description=data.get('description', None),
            confidence_threshold=data.get('confidence_threshold', 0.5),
            source=data.get('source', 'unknown'),
            source_timestamp=data.get('source_timestamp', None),
            merged_from=data.get('merged_from', [])
        )

        for child_data in data.get('children', []):
            child_category = TaxonomyMerger._dict_to_category(child_data)
            category.children.append(child_category)

        return category

    @staticmethod
    def _category_to_dict(category: HierarchicalCategory) -> Dict[str, Any]:
        """Convert a HierarchicalCategory object to a dictionary."""
        data = {
            "name": category.name,
            "description": category.description,
            "confidence_threshold": category.confidence_threshold,
            "children": [TaxonomyMerger._category_to_dict(child) for child in category.children],
            "source": category.source,
            "source_timestamp": category.source_timestamp,
            "merged_from": category.merged_from
        }
        return data

    @staticmethod
    def _flatten_taxonomy(root: HierarchicalCategory, source: str) -> Dict[str, HierarchicalCategory]:
        """Flatten taxonomy into path -> category mapping."""
        categories = {}

        def flatten(category: HierarchicalCategory, current_path: List[str]):
            # Construct the path including the current category
            new_path = current_path + [category.name]
            path_key = " → ".join(new_path)
            
            # Create a copy with source information
            flattened = HierarchicalCategory(
                name=category.name,
                description=category.description,
                confidence_threshold=category.confidence_threshold,
                source=source,
                source_timestamp=category.source_timestamp,
                merged_from=category.merged_from.copy()
            )
            categories[path_key] = flattened

            for child in category.children:
                flatten(child, new_path)

        # Start flattening from root's children to avoid including the root itself in the path keys
        # The root is just a container and shouldn't be part of the merge logic keys
        for child in root.children:
             flatten(child, [])
             
        return categories

    @staticmethod
    def _enrich_category(primary: HierarchicalCategory, secondary: HierarchicalCategory, secondary_source: str):
        """Enrich primary category with information from secondary category."""
        current_time = datetime.now(timezone.utc).isoformat() + "Z"

        # Update source tracking
        if secondary_source not in primary.merged_from:
            primary.merged_from.append(secondary_source)
        primary.source = "merged"
        primary.source_timestamp = current_time

        # Improve description if primary lacks one
        if not primary.description and secondary.description:
            primary.description = secondary.description

        # Use better confidence threshold
        if secondary.confidence_threshold > primary.confidence_threshold:
            primary.confidence_threshold = secondary.confidence_threshold

    @staticmethod
    def _rebuild_hierarchy(root: HierarchicalCategory, category_map: Dict[str, HierarchicalCategory]):
        """Rebuild hierarchical structure from flat category map."""
        # Group categories by their parent path
        path_parts = {}
        for path, category in category_map.items():
            parts = path.split(" → ")
            path_parts[path] = parts

        # Find root level categories (those with just one part)
        root_categories = {}
        for path, parts in path_parts.items():
            if len(parts) == 1:
                root_categories[parts[0]] = category_map[path]

        # Add root categories
        for category in root_categories.values():
            root.children.append(category)

        # Rebuild child relationships
        for path, category in category_map.items():
            if " → " in path:
                parent_path = " → ".join(path.split(" → ")[:-1])
                if parent_path in category_map:
                    parent = category_map[parent_path]
                    # Avoid duplicates
                    if category not in parent.children:
                        parent.children.append(category)