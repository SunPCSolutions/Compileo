"""
Taxonomy Processor

Main entry point for the taxonomy module, orchestrating taxonomy generation,
merging, and loading operations for the document processing pipeline.
"""

import json
from typing import List, Optional
from ...core.logging import get_logger

logger = get_logger(__name__)

from src.compileo.features.extraction.context_models import HierarchicalCategory
from src.compileo.storage.src.project.database_repositories import TaxonomyRepository
from src.compileo.storage.src.project.file_manager import FileManager
from .generator import TaxonomyGenerator
from .loader import TaxonomyLoader
from .merger import TaxonomyMerger


class TaxonomyProcessor:
    """
    Main processor for taxonomy operations in the document processing pipeline.

    This class sits between the chunk module and extraction module, providing:
    - Automatic taxonomy generation from chunks
    - Loading and merging of user-provided taxonomies
    - Intelligent taxonomy merging with source tracking
    - Taxonomy persistence and retrieval
    """

    def __init__(self, taxonomy_repo: TaxonomyRepository,
                 file_manager: FileManager, grok_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None, taxonomy_provider: str = "grok"):
        """
        Initialize TaxonomyProcessor.

        Args:
            processed_output_repo: Repository for taxonomy storage/retrieval
            file_manager: File system manager for taxonomy persistence
            grok_api_key: API key for Grok taxonomy generation (required if using grok)
            gemini_api_key: API key for Gemini taxonomy generation (required if using gemini)
            taxonomy_provider: AI provider for taxonomy generation ("grok", "ollama", or "gemini")
        """
        self.repo = taxonomy_repo
        self.file_manager = file_manager
        self.grok_api_key = grok_api_key
        self.gemini_api_key = gemini_api_key
        self.taxonomy_provider = taxonomy_provider
        from src.compileo.storage.src.project.database_repositories import ProjectRepository
        self.loader = TaxonomyLoader(taxonomy_repo, ProjectRepository(taxonomy_repo.db))

        # Initialize the appropriate generator
        if taxonomy_provider == "ollama":
            from .ollama_generator import OllamaTaxonomyGenerator
            self.generator = OllamaTaxonomyGenerator()
        elif taxonomy_provider == "grok":
            if not grok_api_key:
                raise ValueError("grok_api_key is required when using grok provider")
            self.generator = TaxonomyGenerator(grok_api_key)
        elif taxonomy_provider == "gemini":
            if not gemini_api_key:
                raise ValueError("gemini_api_key is required when using gemini provider")
            from .gemini_generator import GeminiTaxonomyGenerator
            self.generator = GeminiTaxonomyGenerator(gemini_api_key)
        else:
            raise ValueError(f"Unsupported taxonomy provider: {taxonomy_provider}")

    def process_taxonomy(self, chunks: List[str], project_id: int,
                        user_taxonomy: Optional[HierarchicalCategory] = None,
                        auto_generate: bool = True, domain: str = "general",
                        depth: int = 3, sample_size: Optional[int] = None,
                        save_merged: bool = False) -> Optional[HierarchicalCategory]:
        """
        Main taxonomy processing workflow.

        Args:
            chunks: Text chunks from document processing
            project_id: Project identifier for storage
            user_taxonomy: Optional user-provided taxonomy
            auto_generate: Whether to generate taxonomy from chunks
            domain: Knowledge domain for generation
            depth: Maximum taxonomy depth
            sample_size: Number of chunks to sample
            save_merged: Whether to save merged taxonomy

        Returns:
            Processed taxonomy ready for classification
        """
        taxonomy = None

        # Generate taxonomy from chunks if requested
        if auto_generate and chunks:
            try:
                generated_taxonomy_data = self.generator.generate_taxonomy(
                    chunks=chunks,
                    domain=domain,
                    depth=depth,
                    sample_size=sample_size
                )

                # Save generated taxonomy
                filename = f"auto_taxonomy_{domain}_{len(chunks)}chunks.json"
                content = json.dumps(generated_taxonomy_data, indent=2, ensure_ascii=False)
                stored_path = self.file_manager.store_file(project_id, None, filename, content.encode('utf-8'))

                # Store taxonomy metadata in database
                from datetime import datetime
                import uuid
                self.repo.create_taxonomy({
                    "id": str(uuid.uuid4()),
                    "project_id": str(project_id),
                    "name": f"auto_{domain}_{len(chunks)}chunks",
                    "structure": generated_taxonomy_data,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                })

                # Parse back to HierarchicalCategory
                taxonomy_data = generated_taxonomy_data
                if isinstance(taxonomy_data, dict) and "taxonomy" in taxonomy_data:
                    taxonomy = self.loader._parse_taxonomy(taxonomy_data["taxonomy"])
                else:
                    taxonomy = self.loader._parse_taxonomy(taxonomy_data)

                logger.info(f"✅ Auto-generated taxonomy with {len(taxonomy.children)} categories")

            except Exception as e:
                logger.warning(f"Failed to auto-generate taxonomy: {e}")

        # Merge with user taxonomy if both exist
        if user_taxonomy and taxonomy:
            taxonomy = self.loader.merge_taxonomies(user_taxonomy, taxonomy, "manual", "auto")
            logger.info(f"📋 Merged taxonomies: {len(taxonomy.children)} total categories")
        elif user_taxonomy:
            taxonomy = user_taxonomy
            logger.info(f"📄 Using user-provided taxonomy with {len(taxonomy.children)} categories")

        # Save merged taxonomy if requested
        if save_merged and taxonomy:
            try:
                filename = f"merged_taxonomy_{domain}_{len(taxonomy.children)}categories.json"
                taxonomy_dict = taxonomy.to_dict()
                content = json.dumps(taxonomy_dict, indent=2, ensure_ascii=False)
                stored_path = self.file_manager.store_file(project_id, None, filename, content.encode('utf-8'))

                # Store taxonomy metadata in database
                from datetime import datetime
                import uuid
                self.repo.create_taxonomy({
                    "id": str(uuid.uuid4()),
                    "project_id": str(project_id),
                    "name": f"merged_{domain}_{len(taxonomy.children)}categories",
                    "structure": taxonomy_dict,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                })
                logger.info(f"Saved merged taxonomy to: {stored_path}")
            except Exception as e:
                logger.warning(f"Failed to save merged taxonomy: {e}")

        return taxonomy

    def load_project_taxonomy(self, project_name: str,
                            taxonomy_name: Optional[str] = None) -> Optional[HierarchicalCategory]:
        """
        Load taxonomy for a project.

        Args:
            project_name: Name of the project
            taxonomy_name: Specific taxonomy name (None for most recent)

        Returns:
            Loaded taxonomy or None if not found
        """
        return self.loader.load_taxonomy_by_project(project_name, taxonomy_name)

    def load_user_taxonomy(self, filepath: str) -> HierarchicalCategory:
        """
        Load user-provided taxonomy from file.

        Args:
            filepath: Path to taxonomy JSON file

        Returns:
            Loaded taxonomy with manual source marking
        """
        taxonomy = self.loader.load_taxonomy_from_file(filepath)
        TaxonomyLoader._mark_taxonomy_source(taxonomy, "manual")
        return taxonomy

    def validate_taxonomy(self, taxonomy: HierarchicalCategory) -> List[str]:
        """
        Validate taxonomy structure.

        Args:
            taxonomy: Taxonomy to validate

        Returns:
            List of validation errors
        """
        return self.loader.validate_taxonomy(taxonomy)