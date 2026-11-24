"""
Taxonomy Loader

This module handles taxonomy persistence, retrieval, and loading with
project-based storage and source tracking support.
"""

import json
import os
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from src.compileo.features.extraction.context_models import HierarchicalCategory
from src.compileo.storage.src.project.database_repositories import TaxonomyRepository, ProjectRepository


class TaxonomyLoader:
    """Loader for hierarchical taxonomy files using project-structured storage."""

    def __init__(self, taxonomy_repo: TaxonomyRepository, project_repo: Optional[ProjectRepository] = None):
        """
        Initialize TaxonomyLoader with TaxonomyRepository and ProjectRepository.

        Args:
            taxonomy_repo: Repository for accessing taxonomies
            project_repo: Repository for resolving project names to IDs (optional, required for multi-tenant)
        """
        self.taxonomy_repo = taxonomy_repo
        self.project_repo = project_repo
        if not self.project_repo:
            raise ValueError("project_repo is required for multi-tenant architecture")
        self._generation_metadata = None
        self._analytics = None

    def load_taxonomy_by_project(self, project_name: str, owner_id: str, taxonomy_name: Optional[str] = None) -> Optional[HierarchicalCategory]:
        """
        Load taxonomy by project name.

        Args:
            project_name: Name of the project
            owner_id: Owner ID of the project
            taxonomy_name: Specific taxonomy name to load. If None, loads the most recent taxonomy.

        Returns:
            Root HierarchicalCategory object or None if not found
        """
        # Resolve project name to project ID
        project_id = self.project_repo.get_project_by_name(project_name)
        if not project_id:
            return None

        # Get taxonomies for this project
        taxonomies = self.taxonomy_repo.get_taxonomies_by_project(project_id)

        if not taxonomies:
            return None

        if taxonomy_name is None:
            # Load most recent taxonomy (by created_at DESC)
            selected_taxonomy = taxonomies[0]  # Already sorted by created_at DESC
        else:
            # Find taxonomy by name
            selected_taxonomy = None
            for tax in taxonomies:
                if tax.get('name') == taxonomy_name:
                    selected_taxonomy = tax
                    break
            if selected_taxonomy is None:
                return None

        # Parse taxonomy data from database
        # Handle cases where structure might be in 'structure' or 'taxonomy_data' column
        taxonomy_data_str = selected_taxonomy.get('structure') or selected_taxonomy.get('taxonomy_data')
        if not taxonomy_data_str:
            return None

        try:
            data = json.loads(taxonomy_data_str)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid taxonomy data for taxonomy {selected_taxonomy.get('id')}")

        # Handle extended format (with generation metadata)
        if isinstance(data, dict) and "taxonomy" in data:
            taxonomy_data = data["taxonomy"]
            # Store generation metadata for potential future use
            self._generation_metadata = data.get("generation_metadata")
            self._analytics = data.get("analytics")
        else:
            # Legacy format (taxonomy data directly)
            taxonomy_data = data
            self._generation_metadata = None
            self._analytics = None

        return self._parse_taxonomy(taxonomy_data)

    def load_taxonomy_from_file(self, filepath: str) -> Optional[HierarchicalCategory]:
        """
        Load taxonomy from a JSON file (supports both legacy and extended formats).

        Args:
            filepath: Path to the taxonomy JSON file

        Returns:
            Root HierarchicalCategory object or None if invalid
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Taxonomy file not found: {filepath}")

        # Handle extended format (with generation metadata)
        if isinstance(data, dict) and "taxonomy" in data:
            taxonomy_data = data["taxonomy"]
            # Store generation metadata for potential future use
            self._generation_metadata = data.get("generation_metadata")
            self._analytics = data.get("analytics")
        else:
            # Legacy format (taxonomy data directly)
            taxonomy_data = data
            self._generation_metadata = None
            self._analytics = None

        return self._parse_taxonomy(taxonomy_data)

    def merge_taxonomies(self, taxonomy1: HierarchicalCategory, taxonomy2: HierarchicalCategory,
                        source1: str = "manual", source2: str = "auto") -> HierarchicalCategory:
        """
        Merge two taxonomies using TaxonomyMerger service.
        """
        from .merger import TaxonomyMerger
        return TaxonomyMerger.merge_taxonomies(taxonomy1, taxonomy2, source1, source2)

    @staticmethod
    def _mark_taxonomy_source(taxonomy: HierarchicalCategory, source: str):
        """Recursively mark all categories in a taxonomy with their source."""
        current_time = datetime.now(timezone.utc).isoformat() + "Z"

        def mark_recursive(category: HierarchicalCategory):
            category.source = source
            category.source_timestamp = current_time
            for child in category.children:
                mark_recursive(child)

        mark_recursive(taxonomy)

    @staticmethod
    def _parse_taxonomy(data: Dict[str, Any], parent_path: Optional[List[str]] = None) -> Optional[HierarchicalCategory]:
        """Recursively parse taxonomy data."""
        if parent_path is None:
            parent_path = []

        if not isinstance(data, dict):
            return None
            
        name = data.get('name', '').strip()
        if not name:
             # If name is missing, try ID, otherwise skip if no children
             if not data.get('id') and not data.get('children'):
                 return None
             name = data.get('id', 'Unnamed Category')

        category = HierarchicalCategory(
            id=data.get('id'),
            name=name,
            description=data.get('description'),
            parent_path=parent_path,
            confidence_threshold=data.get('confidence_threshold', 0.5),
            source=data.get('source', 'auto'),
            source_timestamp=data.get('source_timestamp'),
            merged_from=data.get('merged_from', [])
        )

        children_data = data.get('children', [])
        if isinstance(children_data, list):
            for child_data in children_data:
                child = TaxonomyLoader._parse_taxonomy(child_data, category.get_full_path())
                if child:
                    category.children.append(child)

        return category

    @staticmethod
    def validate_taxonomy(taxonomy: HierarchicalCategory) -> List[str]:
        """
        Validate taxonomy structure and return any issues found.
        """
        errors = []
        seen_names: set = set()

        def _validate_category(cat: HierarchicalCategory):
            if cat.name in seen_names:
                errors.append(f"Duplicate category name: {cat.name}")
            seen_names.add(cat.name)

            for child in cat.children:
                _validate_category(child)

        _validate_category(taxonomy)
        return errors
