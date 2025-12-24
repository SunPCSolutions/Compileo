"""
Context-aware classification data models.

This module defines the data structures used for context-aware classification,
including document metadata, chunk relationships, and hierarchical categories.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json
import os
from datetime import datetime, timezone


class DocumentContext(BaseModel):
    """Context information about the document being classified."""

    title: Optional[str] = Field(None, description="Document title")
    summary: Optional[str] = Field(None, description="Document summary or abstract")
    author: Optional[str] = Field(None, description="Document author")
    publication_date: Optional[str] = Field(None, description="Publication date")
    source: Optional[str] = Field(None, description="Document source or origin")
    keywords: List[str] = Field(default_factory=list, description="Document keywords or tags")

    model_config = ConfigDict(validate_assignment=True)


class ChunkContext(BaseModel):
    """Context information about adjacent chunks for continuity."""

    previous_chunks: List[str] = Field(default_factory=list, description="Chunks preceding the target chunk")
    next_chunks: List[str] = Field(default_factory=list, description="Chunks following the target chunk")
    chunk_position: Optional[int] = Field(None, description="Position of target chunk in document")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks in document")

    @field_validator('previous_chunks', 'next_chunks')
    @classmethod
    def validate_chunk_content(cls, v):
        """Ensure chunk content is not empty."""
        if v and any(not chunk.strip() for chunk in v):
            raise ValueError("Chunk content cannot be empty or whitespace-only")
        return v


class HierarchicalCategory(BaseModel):
    """Represents a category in a hierarchical taxonomy."""

    name: str = Field(..., description="Category name")
    id: Optional[str] = Field(None, description="Category ID")
    description: Optional[str] = Field(None, description="Category description")
    children: List['HierarchicalCategory'] = Field(default_factory=list, description="Child categories")
    parent_path: List[str] = Field(default_factory=list, description="Path from root to this category")
    confidence_threshold: float = Field(0.5, description="Minimum confidence for this category")

    # Source tracking metadata
    source: str = Field(default="auto", description="Source of this category: 'manual' for user-provided, 'auto' for AI-generated, 'merged' for combined")
    source_timestamp: Optional[str] = Field(default=None, description="When this category was created/added")
    merged_from: List[str] = Field(default_factory=list, description="List of source categories this was merged from")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Ensure category name is valid."""
        if not v or not v.strip():
            raise ValueError("Category name cannot be empty")
        return v.strip()

    @field_validator('confidence_threshold')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v

    def get_full_path(self) -> List[str]:
        """Get the complete hierarchical path from root to this category."""
        return self.parent_path + [self.name]

    def get_full_path_string(self) -> str:
        """Get the complete hierarchical path as a string."""
        return " → ".join(self.get_full_path())

    def find_category(self, path: List[str]) -> Optional['HierarchicalCategory']:
        """Find a category by its hierarchical path."""
        if not path:
            return None

        if path[0] == self.name:
            if len(path) == 1:
                return self
            else:
                for child in self.children:
                    result = child.find_category(path[1:])
                    if result:
                        return result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "children": [child.to_dict() for child in self.children],
            "parent_path": self.parent_path,
            "confidence_threshold": self.confidence_threshold,
            "source": self.source,
            "source_timestamp": self.source_timestamp,
            "merged_from": self.merged_from
        }


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
            id="merged_root",
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
    def _flatten_taxonomy(root: HierarchicalCategory, source: str) -> Dict[str, HierarchicalCategory]:
        """Flatten taxonomy into path -> category mapping."""
        categories = {}

        def flatten(category: HierarchicalCategory, current_path: List[str]):
            path_key = " → ".join(current_path + [category.name])
            # Create a copy with source information
            flattened = HierarchicalCategory(
                id=category.id,
                name=category.name,
                description=category.description,
                confidence_threshold=category.confidence_threshold,
                source=source,
                source_timestamp=category.source_timestamp,
                merged_from=category.merged_from.copy()
            )
            categories[path_key] = flattened

            for child in category.children:
                flatten(child, current_path + [category.name])

        flatten(root, [])
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


class TaxonomyLoader:
    """Loader for hierarchical taxonomy files using project-structured storage."""

    def __init__(self, taxonomy_repo):
        """
        Initialize TaxonomyLoader with TaxonomyRepository.

        Args:
            taxonomy_repo: Repository for accessing taxonomies
        """
        self.repo = taxonomy_repo
        self._generation_metadata = None
        self._analytics = None

    def load_taxonomy_by_project(self, project_name: str, taxonomy_name: Optional[str] = None) -> HierarchicalCategory:
        """
        Load taxonomy by project name.

        Args:
            project_name: Name of the project
            taxonomy_name: Specific taxonomy name to load. If None, loads the most recent taxonomy.

        Returns:
            Root HierarchicalCategory object

        Raises:
            FileNotFoundError: If no taxonomies found for project or specific taxonomy not found
            ValueError: If taxonomy structure is invalid
        """
        taxonomies = self.repo.get_project_outputs_by_type(project_name, "taxonomy")

        if not taxonomies:
            raise FileNotFoundError(f"No taxonomies found for project: {project_name}")

        if taxonomy_name is None:
            # Load most recent taxonomy (by created_at DESC, assuming higher id is more recent if no timestamp sort)
            # Since processed_outputs has created_at, but repo doesn't sort, sort here
            taxonomies.sort(key=lambda x: x[0], reverse=True)  # Assuming id is first column, higher id more recent
            selected_taxonomy = taxonomies[0]
        else:
            # Find taxonomy by name
            selected_taxonomy = None
            for tax in taxonomies:
                filepath = tax[4]  # output_file_path is index 4 (id, project_id, document_id, output_type, output_file_path, created_at)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        tax_data = data.get("taxonomy", data)
                        if tax_data.get("name") == taxonomy_name:
                            selected_taxonomy = tax
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue
            if selected_taxonomy is None:
                raise FileNotFoundError(f"Taxonomy '{taxonomy_name}' not found in project: {project_name}")

        filepath = selected_taxonomy[4]

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

        Args:
            taxonomy1: First taxonomy to merge
            taxonomy2: Second taxonomy to merge
            source1: Source label for first taxonomy
            source2: Source label for second taxonomy

        Returns:
            Merged HierarchicalCategory with source tracking
        """
        return TaxonomyMerger.merge_taxonomies(taxonomy1, taxonomy2, source1, source2)

    @staticmethod
    def _mark_taxonomy_source(taxonomy: HierarchicalCategory, source: str):
        """Recursively mark all categories in a taxonomy with their source."""
        from datetime import datetime
        current_time = datetime.now(timezone.utc).isoformat() + "Z"

        def mark_recursive(category: HierarchicalCategory):
            category.source = source
            category.source_timestamp = current_time
            for child in category.children:
                mark_recursive(child)

        mark_recursive(taxonomy)

    @staticmethod
    def _parse_taxonomy(data: Dict[str, Any], parent_path: Optional[List[str]] = None) -> HierarchicalCategory:
        """Recursively parse taxonomy data."""
        if parent_path is None:
            parent_path = []

        if not isinstance(data, dict) or 'name' not in data:
            raise ValueError("Invalid taxonomy structure: missing 'name' field")

        category = HierarchicalCategory(
            id=data.get('id'),
            name=data['name'],
            description=data.get('description'),
            parent_path=parent_path,
            confidence_threshold=data.get('confidence_threshold', 0.5),
            source=data.get('source', 'auto'),
            source_timestamp=data.get('source_timestamp'),
            merged_from=data.get('merged_from', [])
        )

        children_data = data.get('children', [])
        if not isinstance(children_data, list):
            raise ValueError(f"Invalid children for category '{category.name}': must be a list")

        for child_data in children_data:
            child = TaxonomyLoader._parse_taxonomy(child_data, category.get_full_path())
            category.children.append(child)

        return category

    @staticmethod
    def validate_taxonomy(taxonomy: HierarchicalCategory) -> List[str]:
        """
        Validate taxonomy structure and return any issues found.

        Args:
            taxonomy: Root category to validate

        Returns:
            List of validation error messages
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


class ClassificationResult(BaseModel):
    """Result of context-aware classification."""

    custom: Dict[str, Any] = Field(default_factory=dict, description="User-provided categories with extracted information")
    aigen: Dict[str, Any] = Field(default_factory=dict, description="AI-generated categories with extracted information")
    hierarchy: Dict[str, List[str]] = Field(default_factory=dict, description="Hierarchical category paths")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores for categories")
    document_context_used: bool = Field(False, description="Whether document context was used")
    chunk_context_used: bool = Field(False, description="Whether chunk context was used")

    def add_hierarchical_category(self, category_path: List[str], confidence: float):
        """Add a hierarchical category with confidence score."""
        path_key = " → ".join(category_path)
        self.hierarchy[path_key] = category_path
        self.confidence_scores[path_key] = confidence

    def get_top_categories(self, limit: int = 5) -> List[tuple]:
        """Get top categories by confidence score."""
        sorted_categories = sorted(
            self.confidence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_categories[:limit]