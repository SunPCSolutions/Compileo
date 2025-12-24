"""
Taxonomy Extractor Service

Reusable class for selective content extraction using taxonomy-based classification.
Ports CLI extraction logic into a service with category filtering capabilities.
"""

import json
import os
import logging
import time
import requests
from typing import List, Dict, Any, Optional, Tuple, Set, Union, cast
from datetime import datetime

from src.compileo.features.extraction.multi_stage_classifier import MultiStageClassifier, classify_chunk_multi_stage
from src.compileo.features.extraction.pipeline_config import PipelineConfig, get_default_config
from src.compileo.features.extraction.context_models import HierarchicalCategory
from src.compileo.features.taxonomy.loader import TaxonomyLoader
from src.compileo.storage.src.project.database_repositories import TaxonomyRepository, ChunkRepository
from src.compileo.storage.src.project.file_manager import FileManager
from src.compileo.core.settings import backend_settings
from src.compileo.features.extraction.exceptions import (
    TaxonomyNotFoundError,
    TaxonomyValidationError,
    TaxonomyCategoryError,
    ClassifierUnavailableError,
    ClassificationFailureError,
    ChunkProcessingError,
    FileStorageError,
    DatabaseConnectionError,
    DataValidationError
)
from src.compileo.features.extraction.retry_utils import retry_classification, retry_storage_operation
from src.compileo.features.extraction.error_logging import extraction_logger

# Import from split modules
from .models import ExtractionResult, ExtractionSummary
from .category_resolver import CategoryResolver
from .chunk_loader import ChunkLoader


class TaxonomyExtractor:
    """Service for selective content extraction using taxonomy-based classification."""

    def __init__(self,
                 taxonomy_repo: TaxonomyRepository,
                 chunk_repo: ChunkRepository,
                 file_manager: FileManager,
                 processed_output_repo=None,
                 grok_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 ollama_available: bool = True):
        try:
            # Validate inputs
            if not taxonomy_repo:
                raise ValueError("TaxonomyRepository is required")
            if not chunk_repo:
                raise ValueError("ChunkRepository is required")
            if not file_manager:
                raise ValueError("FileManager is required")

            self.taxonomy_repo = taxonomy_repo
            self.chunk_repo = chunk_repo
            self.file_manager = file_manager
            self.repo = processed_output_repo  # For backward compatibility
            self.grok_api_key = grok_api_key
            self.gemini_api_key = gemini_api_key
            self.openai_api_key = openai_api_key
            self.ollama_available = ollama_available

            # Test database connection
            try:
                # Simple query to test connection
                self.taxonomy_repo.db.cursor().execute("SELECT 1")
            except Exception as e:
                raise DatabaseConnectionError(f"Failed to connect to database: {e}")

            from src.compileo.storage.src.project.database_repositories import ProjectRepository
            self.taxonomy_loader = TaxonomyLoader(taxonomy_repo, ProjectRepository(taxonomy_repo.db))
            self.category_resolver = CategoryResolver()
            self.chunk_loader = ChunkLoader(chunk_repo)
            self.logger = logging.getLogger(__name__)

            # Configure default pipeline config
            self.default_config = get_default_config('balanced')
            if grok_api_key:
                self.default_config.set_api_key('grok', grok_api_key)
            if gemini_api_key:
                self.default_config.set_api_key('gemini', gemini_api_key)
            if openai_api_key:
                self.default_config.set_api_key('openai', openai_api_key)

            # Set classifiers based on availability
            available_classifiers = []
            if grok_api_key:
                available_classifiers.append('grok')
            if gemini_api_key:
                available_classifiers.append('gemini')
            if openai_api_key:
                available_classifiers.append('openai')
            if ollama_available:
                available_classifiers.append('ollama')

            if not available_classifiers:
                raise ClassifierUnavailableError(
                    "At least one classifier must be available (grok, gemini, or ollama)"
                )

            self.default_config.classifiers = available_classifiers

            extraction_logger.log_operation_start(
                "taxonomy_extractor_initialization",
                context={"available_classifiers": available_classifiers}
            )

        except Exception as e:
            extraction_logger.log_error(
                e,
                "taxonomy_extractor_initialization",
                context={"grok_key_available": grok_api_key is not None,
                        "gemini_key_available": gemini_api_key is not None,
                        "openai_key_available": openai_api_key is not None,
                        "ollama_available": ollama_available}
            )
            raise

    def _expand_categories_with_parents(self, taxonomy: HierarchicalCategory, selected_categories: List[str]) -> Set[str]:
        """
        Expand selected category IDs to include all parent categories.

        Args:
            taxonomy: The taxonomy hierarchy
            selected_categories: List of originally selected category IDs

        Returns:
            Set of expanded category IDs including parents
        """
        expanded_ids = set(selected_categories)

        # Build a map of all categories by ID for quick lookup
        category_map = {}

        def build_category_map(node: HierarchicalCategory):
            if hasattr(node, 'id') and node.id:
                category_map[node.id] = node
            if node.children:
                for child in node.children:
                    build_category_map(child)

        build_category_map(taxonomy)

        for cat_id in selected_categories:
            if cat_id in category_map:
                category = category_map[cat_id]
                # Use parent_path to find parent IDs
                # parent_path contains the path from root to parent, so we need to find the IDs
                for parent_name in category.parent_path:
                    # Find the category with this name in the map
                    for cat_id_in_map, cat_obj in category_map.items():
                        if cat_obj.name == parent_name and cat_id_in_map not in expanded_ids:
                            expanded_ids.add(cat_id_in_map)
                            break

        return expanded_ids


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

    def extract_content(self,
                           project_id: Union[str, int],
                           taxonomy_project: str,
                           taxonomy_name: Optional[str] = None,
                           selected_categories: Optional[List[str]] = None,
                           confidence_threshold: float = 0.5,
                           max_chunks: Optional[int] = None,
                           pipeline_config: Optional[PipelineConfig] = None,
                           taxonomy_data: Optional[HierarchicalCategory] = None,
                           initial_classifier: Optional[str] = None,
                           extraction_type: str = "ner",
                           enable_validation_stage: bool = False,
                           validation_classifier: Optional[str] = None,
                           extraction_mode: str = "contextual") -> Tuple[List[ExtractionResult], ExtractionSummary]:
        """Extract content from chunks using taxonomy-based classification."""
        start_time = time.time()

        # Type safe logging wrapper
        def safe_log_start(op_name: str, ctx: Dict[str, Any], pid: Union[str, int, None] = None):
            pid_int = None
            if isinstance(pid, int):
                pid_int = pid
            elif isinstance(pid, str) and pid.isdigit():
                pid_int = int(pid)
            
            extraction_logger.log_operation_start(
                op_name,
                context=ctx,
                project_id=pid_int
            )

        def safe_log_warning(msg: str, op_name: str, ctx: Dict[str, Any], pid: Union[str, int, None] = None):
            pid_int = None
            if isinstance(pid, int):
                pid_int = pid
            elif isinstance(pid, str) and pid.isdigit():
                pid_int = int(pid)
            
            extraction_logger.log_warning(
                msg,
                op_name,
                context=ctx,
                project_id=pid_int
            )

        safe_log_start(
            "extract_content",
            {
                "project_id": project_id,
                "taxonomy_project": taxonomy_project,
                "taxonomy_name": taxonomy_name,
                "selected_categories": selected_categories,
                "confidence_threshold": confidence_threshold,
                "max_chunks": max_chunks,
                "initial_classifier": initial_classifier,
                "extraction_type": extraction_type,
                "enable_validation_stage": enable_validation_stage,
                "validation_classifier": validation_classifier,
                "extraction_mode": extraction_mode
            },
            project_id
        )

        try:
            # Validate inputs
            if not project_id or not isinstance(project_id, (str, int)):
                raise ValueError("Invalid project_id: must be a valid identifier")
            if not taxonomy_project and not taxonomy_data:
                raise ValueError("Either taxonomy_project or taxonomy_data must be provided.")
            if confidence_threshold < 0.0 or confidence_threshold > 1.0:
                raise ValueError("Invalid confidence_threshold: must be between 0.0 and 1.0")
            if extraction_mode not in ["contextual", "document_wide"]:
                raise ValueError("Invalid extraction_mode: must be 'contextual' or 'document_wide'")

            # Load taxonomy with error handling
            if taxonomy_data:
                taxonomy = taxonomy_data
            else:
                try:
                    taxonomy = self.taxonomy_loader.load_taxonomy_by_project(taxonomy_project, taxonomy_name)
                    if not taxonomy:
                        raise TaxonomyNotFoundError(taxonomy_project, taxonomy_name)
                except Exception as e:
                    if isinstance(e, TaxonomyNotFoundError):
                        raise
                    raise TaxonomyValidationError(f"Failed to load taxonomy: {e}", taxonomy_project)

            # Validate taxonomy structure
            if not hasattr(taxonomy, 'children') or not taxonomy.children:
                raise TaxonomyValidationError("Taxonomy has no categories", taxonomy_project)

            # For selective extraction, use selected_categories directly (ensuring string types for safe comparison)
            target_category_ids = {str(cid) for cid in (selected_categories or [])}
            safe_log_start(
                "selective_extraction_setup",
                {
                    "selected_category_ids": list(target_category_ids),
                    "category_count": len(target_category_ids),
                    "raw_types": [type(cid).__name__ for cid in (selected_categories or [])[:10]]
                },
                project_id
            )

            # Determine categories for classification
            if target_category_ids:
                # If categories are selected, classify directly against them
                classification_categories = []
                for cat_id in target_category_ids:
                    cat_info = self.category_resolver.get_category_info_by_id(taxonomy, cat_id)
                    if cat_info and cat_info.get('name'):
                        classification_categories.append(cat_info['name'])

                if not classification_categories:
                    raise TaxonomyCategoryError("Could not resolve any selected category IDs to names.", str(target_category_ids))
            else:
                # Use the children of the root as the high-level categories for classification
                if hasattr(taxonomy, 'children') and taxonomy.children:
                    classification_categories = [child.name for child in taxonomy.children]
                else:
                    classification_categories = []

            if not classification_categories:
                raise TaxonomyValidationError("Taxonomy has no categories to use for classification", taxonomy_project)

            # Load chunks with error handling
            try:
                chunks_data = self.chunk_loader.load_project_chunks(project_id, max_chunks)
            except Exception as e:
                raise ChunkProcessingError(f"Failed to load project chunks: {e}")

            if not chunks_data:
                raise ChunkProcessingError(f"No chunks found for project {project_id}")

            # Initialize classifier with retry logic
            config = pipeline_config or self.default_config
            try:
                classifier = MultiStageClassifier(config=config)
            except Exception as e:
                raise ClassificationFailureError(f"Failed to initialize classifier: {e}")

            # Process chunks with comprehensive error handling
            results = []
            processed_count = 0
            failed_chunks = 0
            
            # Log extraction start summary
            safe_log_start(
                "entity_extraction_batch_started",
                {
                    "total_chunks": len(chunks_data),
                    "selected_categories": list(target_category_ids),
                    "initial_classifier": initial_classifier
                },
                project_id
            )

            # Pre-process: Group target categories by their parent contexts
            # Removed redundant context_map logic - will be handled per chunk or per extraction pass

            for i, chunk_data in enumerate(chunks_data):
                chunk_start_time = time.time()

                try:
                    chunk_id = chunk_data.get('id')
                    chunk_text = chunk_data.get('text')

                    if not chunk_id or not chunk_text or not isinstance(chunk_text, str):
                        safe_log_warning(
                            "Invalid chunk data",
                            "extract_content",
                            {"chunk_data_keys": list(chunk_data.keys()), "chunk_text_type": type(chunk_text).__name__},
                            project_id
                        )
                        continue

                    # Ensure chunk_text is a string
                    chunk_text = str(chunk_text).strip()
                    if not chunk_text:
                        continue

                    # Log progress periodically (every 10 chunks) instead of per-chunk logging
                    if i > 0 and i % 10 == 0:
                        safe_log_start(
                            "entity_extraction_progress",
                            {
                                "chunks_processed": i,
                                "total_chunks": len(chunks_data),
                                "selected_categories": list(target_category_ids),
                                "initial_classifier": initial_classifier
                            },
                            project_id
                        )

                    # Build parent-child structure for contextual extraction
                    parent_child_structure = {}
                    for cat_id in target_category_ids:
                        # Find immediate parent category for context
                        current_category = None
                        for node in self._traverse_taxonomy(taxonomy):
                            if hasattr(node, 'id') and node.id == cat_id:
                                current_category = node
                                break

                        if current_category and hasattr(current_category, 'parent_path') and current_category.parent_path:
                            immediate_parent_name = current_category.parent_path[-1]
                            found_parent = False
                            for node in self._traverse_taxonomy(taxonomy):
                                # Check if node matches the parent name
                                if hasattr(node, 'name') and node.name == immediate_parent_name:
                                    node_id_str = str(node.id) if hasattr(node, 'id') else None
                                    
                                    # LOGGING: Check if we are about to skip a parent context because it is also a target
                                    if node_id_str and node_id_str in target_category_ids:
                                        extraction_logger.log_warning(
                                            f"Category {cat_id} ('{current_category.name}') has a parent '{node.name}' (ID: {node_id_str}) that is also in target_category_ids. "
                                            "Previously, this would cause the child to be OMITTED from the prompt context.",
                                            "parent_context_debug",
                                            context={"child_id": cat_id, "parent_id": node_id_str, "in_targets": True},
                                            project_id=int(project_id) if isinstance(project_id, (str, int)) and str(project_id).isdigit() else None
                                        )

                                    if node_id_str:
                                        parent_cat_info = self.category_resolver.get_category_info_by_id(taxonomy, node_id_str)
                                        if parent_cat_info and parent_cat_info.get('name'):
                                            if node_id_str not in parent_child_structure:
                                                parent_child_structure[node_id_str] = {
                                                    'name': parent_cat_info['name'],
                                                    'description': parent_cat_info.get('description', ''),
                                                    'children': []
                                                }
                                            child_info = self.category_resolver.get_category_info_by_id(taxonomy, cat_id)
                                            if child_info and child_info.get('name'):
                                                parent_child_structure[node_id_str]['children'].append({
                                                    'id': cat_id,
                                                    'name': child_info['name'],
                                                    'description': child_info.get('description', '')
                                                })
                                            found_parent = True
                                            break
                            
                            if not found_parent:
                                extraction_logger.log_warning(
                                    f"Could not find parent node for {cat_id} with name '{immediate_parent_name}'",
                                    "parent_node_not_found",
                                    {"child_id": cat_id, "parent_name": immediate_parent_name},
                                    project_id=int(project_id) if isinstance(project_id, (str, int)) and str(project_id).isdigit() else None
                                )
                        else:
                            # Root level category - treat as its own parent
                            cat_info = self.category_resolver.get_category_info_by_id(taxonomy, cat_id)
                            if cat_info and cat_info.get('name'):
                                parent_id = str(cat_id)
                                parent_child_structure[parent_id] = {
                                    'name': cat_info['name'],
                                    'description': cat_info.get('description', ''),
                                    'children': [{
                                        'id': cat_id,
                                        'name': cat_info['name'],
                                        'description': cat_info.get('description', '')
                                    }]
                                }

                    # All selected categories will be processed - context relevance handled in prompt
                    active_target_ids = list(target_category_ids)

                    extracted_entities = {}
                    entity_confidences = {}

                    # Build category information for the prompt
                    category_info = []

                    for cat_id in target_category_ids:
                        cat_info = self.category_resolver.get_category_info_by_id(taxonomy, cat_id)
                        if cat_info and cat_info.get('name'):
                            category_info.append({
                                'id': cat_id,
                                'name': cat_info['name'],
                                'description': cat_info.get('description', ''),
                                'is_selected': True
                            })

                            # For contextual mode, include immediate parent categories for context
                            if extraction_mode == "contextual":
                                # Find immediate parent category for context
                                current_category = None
                                for node in self._traverse_taxonomy(taxonomy):
                                    if hasattr(node, 'id') and node.id == cat_id:
                                        current_category = node
                                        break

                                if current_category and hasattr(current_category, 'parent_path') and current_category.parent_path:
                                    # Only include the immediate parent (last item in parent_path)
                                    immediate_parent_name = current_category.parent_path[-1]
                                    for node in self._traverse_taxonomy(taxonomy):
                                        if (hasattr(node, 'name') and node.name == immediate_parent_name and
                                            hasattr(node, 'id') and node.id and node.id not in target_category_ids):
                                            parent_cat_info = self.category_resolver.get_category_info_by_id(taxonomy, node.id)
                                            if parent_cat_info and parent_cat_info.get('name'):
                                                category_info.append({
                                                    'id': node.id,
                                                    'name': parent_cat_info['name'],
                                                    'is_selected': False,
                                                    'is_parent': True
                                                })
                                                break

                    if not category_info:
                        safe_log_warning(
                            "No valid categories found for extraction",
                            "entity_extraction_no_categories",
                            {"chunk_id": chunk_id, "target_category_ids": list(target_category_ids)},
                            project_id
                        )
                        continue

                    # Create parent-child structured prompt for contextual extraction
                    # For document_wide mode, skip parent-child structure filtering
                    parent_child_text = ""
                    if extraction_mode == "contextual":
                        for parent_id, parent_data in parent_child_structure.items():
                            parent_child_text += f"\nParent: {parent_data['name']} (ID: {parent_id})\n"
                            if parent_data.get('description'):
                                parent_child_text += f"  Description: {parent_data['description']}\n"
                            parent_child_text += "  Children:\n"
                            for child in parent_data['children']:
                                parent_child_text += f"    - {child['name']} (ID: {child['id']})\n"
                                if child.get('description'):
                                    parent_child_text += f"      Description: {child['description']}\n"
                    else:
                        # For document_wide, create a focused list of selected categories only
                        parent_child_text = "Selected categories to extract from (independent of parent relationships):\n"
                        for cat_info in category_info:
                            if cat_info.get('is_selected', False):
                                parent_child_text += f"ID: {cat_info['id']} - Name: {cat_info['name']}\n"
                                if cat_info.get('description'):
                                    parent_child_text += f"      Description: {cat_info['description']}\n"
                    # Enhanced prompt with role-playing and chain-of-thought for Ollama
                    # Ensure we use the same effective classifier as context check
                    effective_classifier = initial_classifier or backend_settings.get_classification_model() or 'grok'

                    # Initialize entity_prompt to avoid scoping issues
                    entity_prompt = ""

                    if extraction_type == "whole_text":
                        if extraction_mode == "document_wide":
                            # Document-wide extraction: process all chunks for selected categories without contextual filtering
                            if effective_classifier == 'ollama':
                                entity_prompt = f"""You are an expert content analyst with extensive experience in document analysis and information extraction. Your task is to identify and extract complete text portions from the provided content that discuss each of the specified categories.

CRITICAL DOCUMENT-WIDE RULES:
- Process ALL specified categories regardless of content context or topic relevance
- Do not filter or skip categories based on whether their parent topics are discussed
- Extract text portions for every category in the list, even if the text doesn't seem related
- If no text portions are found for a category, return an empty array (don't skip the category)

Now, follow this step-by-step reasoning process:

1. **Read and Understand**: Carefully read the entire text to understand its context and content.

2. **Category Extraction**: For EACH AND EVERY specified category, search for complete text portions that discuss it.

3. **Text Portion Validation**: For each potential text portion found:
    - Verify it contains meaningful information about the category
    - Ensure it forms a complete, coherent unit of text

4. **Confidence Assessment**: Rate your confidence in each extraction based on:
    - How directly the text portion addresses the category
    - The completeness and coherence of the extracted text

5. **Final Review**: Double-check that all specified categories are included in the response, even if they have empty text arrays.

IMPORTANT RULES:
- Process ALL categories in the list - do not skip any
- Extract COMPLETE text portions (paragraphs, sentences, or coherent sections) - not just keywords
- Preserve the original text exactly as written - no summarization or modification
- Only extract text that is EXPLICITLY present in the provided content
- Focus on substantial, informative text portions rather than brief mentions
- Keep in mind that the content can be from any domain, so adapt your analysis accordingly

Text to analyze:
{chunk_text}

Return your response as a JSON object where each key is a category ID and each value is an object with:
- "texts": array of strings (the complete text portions found for this category)
- "confidence": number between 0 and 1 (your confidence in the extraction)

Example response format:
{{
  "category_id_1": {{
    "texts": ["Complete text portion discussing category 1 in detail..."],
    "confidence": 0.85
  }},
  "category_id_2": {{
    "texts": [],
    "confidence": 0.0
  }}
}}

JSON Response:"""
                            else:
                                # Document-wide prompt for other classifiers (Grok, Gemini)
                                entity_prompt = f"""Extract complete text portions from the following content for each of the specified categories.

{parent_child_text}

CRITICAL DOCUMENT-WIDE RULES:
- Process ALL specified categories regardless of content context or topic relevance.
- Do not filter or skip categories based on whether their parent topics are discussed.
- Extract text portions for every category in the list, even if the text doesn't seem related.
- If no text portions are found for a category, return an empty array (don't skip the category).
- YOU MUST USE THE EXACT CATEGORY ID AS THE KEY IN THE JSON RESPONSE. DO NOT USE THE CATEGORY NAME.
- DO NOT INCLUDE PARENT IDs IN THE JSON RESPONSE UNLESS THEY ARE ALSO LISTED AS TARGET CATEGORIES.

Return your response as a JSON object where each key is the exact category ID provided above, and each value is an object with:
- "texts": array of strings (the complete text portions found for this category)
- "confidence": number between 0 and 1 (your confidence in the extraction)

Example response format:
{{
  "1100": {{
    "texts": ["Complete text portion discussing category..."],
    "confidence": 0.85
  }},
  "1120": {{
    "texts": [],
    "confidence": 0.0
  }}
}}

Rules:
- USE CATEGORY IDs AS KEYS (e.g., "1100", "1120"), NOT NAMES.
- Process ALL categories - do not skip any based on context.
- Extract complete, coherent text portions (not just keywords).
- Preserve original text exactly as written.
- Only include text that is explicitly present in the content.
- Focus on substantial, informative text portions.

Text to analyze:
{chunk_text}

JSON Response:"""
                        else:
                            # Contextual extraction: filter by parent-child relationships
                            if effective_classifier == 'ollama':
                                entity_prompt = f"""You are an expert content analyst with extensive experience in document analysis and information extraction. Your task is to identify and extract complete text portions from the provided content that discuss each of the specified child categories, but ONLY if the text is relevant to their parent category.

{parent_child_text}

CRITICAL CONTEXTUAL RULES:
- For each child category, first determine if the text discusses the parent category (explicitly or implied through related concepts)
- Only extract text portions for child categories if the parent context is present in the text
- If the parent category is not discussed in the text, return empty arrays for all its children
- Consider implied context (e.g., symptoms described in a section about Metabolic Syndrome imply the parent topic)

Now, follow this step-by-step reasoning process:

1. **Read and Understand**: Carefully read the entire text to understand its context and content.

2. **Parent Context Analysis**: For each parent category, determine if the text discusses it (explicitly or implied).

3. **Child Category Extraction**: For parents that are relevant, extract complete text portions for their child categories.

4. **Text Portion Validation**: For each potential text portion found:
    - Verify it contains meaningful information about the child category
    - Ensure it forms a complete, coherent unit of text
    - Confirm the parent context is present in the text

5. **Confidence Assessment**: Rate your confidence in each extraction based on:
    - How directly the text portion addresses the child category
    - The completeness and coherence of the extracted text
    - Whether the parent context is clearly present

6. **Final Review**: Double-check that all extracted text portions are actually present in the text and that their parent contexts are relevant.

IMPORTANT RULES:
- Extract COMPLETE text portions (paragraphs, sentences, or coherent sections) - not just keywords
- Preserve the original text exactly as written - no summarization or modification
- Only extract text that is EXPLICITLY present in the provided content
- If a parent category is not relevant to the text, use empty arrays for all its children
- Focus on substantial, informative text portions rather than brief mentions
- Keep in mind that the content can be from any domain, so adapt your analysis accordingly

Text to analyze:
{chunk_text}

Return your response as a JSON object where each key is a category ID and each value is an object with:
- "texts": array of strings (the complete text portions found for this category)
- "confidence": number between 0 and 1 (your confidence in the extraction)

Example response format:
{{
  "category_id_1": {{
    "texts": ["Complete text portion discussing category 1 in detail..."],
    "confidence": 0.85
  }},
  "category_id_2": {{
    "texts": [],
    "confidence": 0.0
  }}
}}

JSON Response:"""
                            else:
                                # Whole text prompt for other classifiers (Grok, Gemini) with contextual structure
                                entity_prompt = f"""Extract complete text portions from the following content for each of the specified child categories, but ONLY if the text is relevant to their parent category.

{parent_child_text}

Contextual Rules:
- Only extract for child categories if their parent category is discussed in the text (explicitly or implied).
- If parent category is not present, return empty arrays for children.
- Consider implied context (e.g., symptoms implying Metabolic Syndrome discussion).
- YOU MUST USE THE EXACT CATEGORY ID AS THE KEY IN THE JSON RESPONSE. DO NOT USE THE CATEGORY NAME.
- DO NOT INCLUDE PARENT IDs IN THE JSON RESPONSE UNLESS THEY ARE ALSO LISTED AS CHILD CATEGORIES.

Return your response as a JSON object where each key is a category ID and each value is an object with:
- "texts": array of strings (the complete text portions found for this category)
- "confidence": number between 0 and 1 (your confidence in the extraction)

Example response format:
{{
  "11001": {{
    "texts": ["Complete text portion discussing category 1..."],
    "confidence": 0.85
  }},
  "11002": {{
    "texts": [],
    "confidence": 0.0
  }}
}}

Rules:
- USE CATEGORY IDs AS KEYS (e.g., "11001", "11002"), NOT NAMES.
- Extract complete, coherent text portions (not just keywords).
- Preserve original text exactly as written.
- Only include text that is explicitly present in the content.
- If parent context missing, use empty array for "texts".
- Focus on substantial, informative text portions.

Text to analyze:
{chunk_text}

JSON Response:"""
                                # DEBUG: Log the actual prompt being sent for document_wide whole_text
                                safe_log_start(
                                    "document_wide_prompt_debug",
                                    {
                                        "extraction_mode": extraction_mode,
                                        "extraction_type": extraction_type,
                                        "effective_classifier": effective_classifier,
                                        "parent_child_text_preview": parent_child_text[:200] if parent_child_text else "EMPTY",
                                        "chunk_id": chunk_id
                                    },
                                    project_id
                                )
                    else:
                        # NER extraction prompts
                        if extraction_mode == "document_wide":
                            # Document-wide NER extraction: process all chunks for selected categories without contextual filtering
                            if effective_classifier == 'ollama':
                                entity_prompt = f"""You are an expert information extraction specialist with years of experience in content analysis and taxonomy classification. Your task is to carefully analyze the provided text and extract specific entities that belong to each of the given categories.

{parent_child_text}

CRITICAL DOCUMENT-WIDE RULES:
- Process ALL specified categories regardless of content context or topic relevance
- Do not filter or skip categories based on whether their parent topics are discussed
- Extract entities for every category in the list, even if the text doesn't seem related
- If no entities are found for a category, return an empty array (don't skip the category)

Now, follow this step-by-step reasoning process:

1. **Read and Understand**: Carefully read the entire text to understand its context and content.

2. **Category Extraction**: For EACH AND EVERY specified category, search for entities that belong to it.

3. **Entity Validation**: For each potential entity found:
    - Verify it is explicitly mentioned in the text
    - Ensure it logically belongs to the category

4. **Confidence Assessment**: Rate your confidence in each extraction based on:
    - How clearly the entity is mentioned
    - How well it fits the category

5. **Final Review**: Double-check that all specified categories are included in the response, even if they have empty entity arrays.

IMPORTANT RULES:
- Process ALL categories in the list - do not skip any
- Only extract entities that are EXPLICITLY MENTIONED in the text
- Do not infer, assume, or add entities that aren't directly stated
- Do not use generic terms - only specific, named entities
- Be precise and avoid over-extraction
- Keep in mind that the text can be from any domain (e.g., legal, financial, technical, general knowledge), so adapt your analysis accordingly.

Text to analyze:
{chunk_text}

Return your response as a JSON object where each key is a category ID and each value is an object with:
- "entities": array of strings (the specific entities found for this category)
- "confidence": number between 0 and 1 (your confidence in the extraction)

Example response format:
{{
  "category_id_1": {{
    "entities": ["entity1", "entity2"],
    "confidence": 0.8
  }},
  "category_id_2": {{
    "entities": [],
    "confidence": 0.0
  }}
}}

JSON Response:"""
                            else:
                                # Document-wide NER prompt for other classifiers (Grok, Gemini) with examples
                                entity_prompt = f"""Extract entities from the following text for each of the specified categories.

{parent_child_text}

CRITICAL DOCUMENT-WIDE RULES:
- Process ALL specified categories regardless of content context or topic relevance
- Do not filter or skip categories based on whether their parent topics are discussed
- Extract entities for every category in the list, even if the text doesn't seem related
- If no entities are found for a category, return an empty array (don't skip the category)

Step-by-step reasoning:
1. Read the entire text carefully
2. For EACH category independently:
   - Search for specific entities that match the category (e.g., for "Surgical & Procedural Treatments", extract names of surgeries, procedures, or treatments mentioned)
   - Validate they are explicitly mentioned
   - Assess confidence
3. Include ALL categories in response, even with empty arrays

Rules:
- Process ALL categories - do not skip any based on context
- Only include entities that are explicitly mentioned in the text
- Be specific and accurate in your extractions
- Do not include any explanations or additional text outside the JSON
- Examples of entities:
  - For "Surgical & Procedural Treatments": "Hysterectomy", "Uterine artery embolization", "Radiofrequency ablation"
  - For "Clinical Conditions": "Adenomyosis", "Leiomyomas", "Endometriosis"

Text to analyze:
{chunk_text}

Return your response as a JSON object where each key is a category ID and each value is an object with:
- "entities": array of strings (the specific entities found for this category)
- "confidence": number between 0 and 1 (your confidence in the extraction)

Example response format:
{{
  "11000": {{
    "entities": ["Ischemic Heart Disease", "Acute Coronary Syndrome"],
    "confidence": 0.9
  }},
  "11111": {{
    "entities": ["Hysterectomy", "Uterine Artery Embolization"],
    "confidence": 0.85
  }},
  "11201": {{
    "entities": [],
    "confidence": 0.0
  }}
}}

JSON Response:"""
                        else:
                            # Contextual NER extraction: filter by parent-child relationships
                            if effective_classifier == 'ollama':
                                entity_prompt = f"""You are an expert information extraction specialist with years of experience in content analysis and taxonomy classification. Your task is to carefully analyze the provided text and extract specific entities that belong to each of the given child categories, but ONLY if the text is relevant to their parent category.

{parent_child_text}

CRITICAL CONTEXTUAL RULES:
- For each child category, first determine if the text discusses the parent category (explicitly or implied through related concepts)
- Only extract entities for child categories if the parent context is present in the text
- If the parent category is not discussed in the text, return empty arrays for all its children
- Consider implied context (e.g., symptoms described in a section about Metabolic Syndrome imply the parent topic)

Now, follow this step-by-step reasoning process:

1. **Read and Understand**: Carefully read the entire text to understand its context and content.

2. **Parent Context Analysis**: For each parent category, determine if the text discusses it (explicitly or implied).

3. **Child Category Extraction**: For parents that are relevant, extract entities for their child categories.

4. **Entity Validation**: For each potential entity found:
    - Verify it is explicitly mentioned in the text
    - Ensure it logically belongs to the child category
    - Confirm the parent context is present in the text

5. **Confidence Assessment**: Rate your confidence in each extraction based on:
    - How clearly the entity is mentioned
    - How well it fits the child category
    - Whether the parent context is clearly present

6. **Final Review**: Double-check that all extracted entities are actually present in the text and that their parent contexts are relevant.

IMPORTANT RULES:
- Only extract entities that are EXPLICITLY MENTIONED in the text
- Do not infer, assume, or add entities that aren't directly stated
- Do not use generic terms - only specific, named entities
- If a parent category is not relevant to the text, use empty arrays for all its children
- Be precise and avoid over-extraction
- Keep in mind that the text can be from any domain (e.g., legal, financial, technical, general knowledge), so adapt your analysis accordingly.

Text to analyze:
{chunk_text}

Return your response as a JSON object where each key is a category ID and each value is an object with:
- "entities": array of strings (the specific entities found for this category)
- "confidence": number between 0 and 1 (your confidence in the extraction)

Example response format:
{{
  "category_id_1": {{
    "entities": ["entity1", "entity2"],
    "confidence": 0.8
  }},
  "category_id_2": {{
    "entities": [],
    "confidence": 0.0
  }}
}}

JSON Response:"""
                            else:
                                # Original NER prompt for other classifiers (Grok, Gemini) with contextual structure
                                entity_prompt = f"""Extract entities from the following text for each of the specified child categories, but ONLY if the text is relevant to their parent category.

{parent_child_text}

Contextual Rules:
- Only extract for child categories if their parent category is discussed in the text (explicitly or implied).
- If parent category is not present, return empty arrays for children.
- Consider implied context (e.g., symptoms implying Metabolic Syndrome discussion).
- YOU MUST USE THE EXACT CATEGORY ID AS THE KEY IN THE JSON RESPONSE. DO NOT USE THE CATEGORY NAME.
- DO NOT INCLUDE PARENT IDs IN THE JSON RESPONSE UNLESS THEY ARE ALSO LISTED AS CHILD CATEGORIES.

Return your response as a JSON object where each key is a category ID and each value is an object with:
- "entities": array of strings (the specific entities found for this category)
- "confidence": number between 0 and 1 (your confidence in the extraction)

Example response format:
{{
  "11001": {{
    "entities": ["entity1", "entity2"],
    "confidence": 0.8
  }},
  "11002": {{
    "entities": [],
    "confidence": 0.0
  }}
}}

Rules:
- USE CATEGORY IDs AS KEYS (e.g., "11001", "11002"), NOT NAMES.
- Only include entities that are explicitly mentioned in the text.
- If parent context missing, use empty array for "entities".
- Be specific and accurate in your extractions.
- Do not include any explanations or additional text outside the JSON.

Text to analyze:
{chunk_text}

JSON Response:"""

                    # Extract entities for all categories in a single API call
                    # Access backend_settings before defining the nested function to avoid scoping issues
                    grok_model = backend_settings.get_classification_grok_model() if effective_classifier == 'grok' else None
                    gemini_model = backend_settings.get_classification_gemini_model() if effective_classifier == 'gemini' else None
                    openai_model = backend_settings.get_classification_openai_model() if effective_classifier == 'openai' else None
                    ollama_model = backend_settings.get_classification_ollama_model() if effective_classifier == 'ollama' else None
                    ollama_base_url = backend_settings.get_ollama_base_url() if effective_classifier == 'ollama' else None
                    ollama_num_ctx = backend_settings.get_classification_ollama_num_ctx() if effective_classifier == 'ollama' else None
                    ollama_temperature = backend_settings.get_classification_ollama_temperature() if effective_classifier == 'ollama' else None
                    ollama_repeat_penalty = backend_settings.get_classification_ollama_repeat_penalty() if effective_classifier == 'ollama' else None
                    ollama_top_p = backend_settings.get_classification_ollama_top_p() if effective_classifier == 'ollama' else None
                    ollama_top_k = backend_settings.get_classification_ollama_top_k() if effective_classifier == 'ollama' else None
                    ollama_num_predict = backend_settings.get_classification_ollama_num_predict() if effective_classifier == 'ollama' else None
                    ollama_seed = backend_settings.get_classification_ollama_seed() if effective_classifier == 'ollama' else None

                    @retry_classification(logger=extraction_logger.logger)
                    def extract_all_entities_with_retry():
                        try:
                            # Move import outside the function if not already available
                            try:
                                from google import genai # type: ignore
                            except ImportError:
                                pass # Handled inside the function logic

                            # Use direct API call for batch entity extraction
                            try:
                                from ...features.extraction.grok_classifier import _sanitize_json

                                if effective_classifier == 'grok':
                                    api_key = self.grok_api_key or ""
                                    # Use pre-fetched grok_model from outer scope

                                    api_url = "https://api.x.ai/v1/chat/completions"
                                    request_data = {
                                        "model": grok_model,
                                        "messages": [
                                            {"role": "system", "content": "You are a precise data extraction assistant. You must output ONLY valid JSON. Do not include any conversational text, markdown formatting, or explanations."},
                                            {"role": "user", "content": entity_prompt}
                                        ],
                                        "temperature": 0.1
                                    }
                                    headers = {
                                        "Authorization": f"Bearer {api_key}",
                                        "Content-Type": "application/json"
                                    }

                                    response = requests.post(api_url, json=request_data, headers=headers, timeout=180)  # Increased timeout to 180s
                                    response.raise_for_status()

                                    response_json = response.json()
                                    response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "{}")

                                    # Reduced debug logging - only log API errors, not every response

                                    # Try to extract JSON from response
                                    sanitized_text = ""
                                    try:
                                        sanitized_text = _sanitize_json(response_text)
                                        batch_result = json.loads(sanitized_text)
                                        
                                        # DEBUG: Log the batch result keys to see what we got
                                        if isinstance(batch_result, dict):
                                            # Check if our target category IDs are in the keys
                                            target_ids = [c['id'] for c in category_info if c.get('is_selected')]
                                            missing = [tid for tid in target_ids if tid not in batch_result]
                                            return batch_result
                                        else:
                                            pass
                                    except Exception as parse_error:
                                        # Log the raw response causing the error
                                        extraction_logger.log_error(
                                            parse_error,
                                            "batch_grok_json_parse_debug",
                                            context={
                                                "response_text_length": len(response_text),
                                                "response_preview": response_text[:1000]
                                            }
                                        )
                                        # Log the raw response causing the error
                                        extraction_logger.log_error(
                                            parse_error,
                                            "batch_grok_json_parse_debug",
                                            context={
                                                "response_text_length": len(response_text),
                                                "response_preview": response_text[:1000]
                                            }
                                        )
                                        extraction_logger.log_warning(
                                            f"Batch JSON parsing failed: {parse_error}",
                                            "batch_grok_json_parse_error",
                                            context={
                                                "response_text": response_text,
                                                "sanitized_text": sanitized_text if sanitized_text else "N/A"
                                            }
                                        )
                                        return {}

                                elif effective_classifier == 'gemini':
                                    api_key = self.gemini_api_key
                                    if api_key:
                                        sanitized_text = "N/A"  # Initialize for error logging
                                        genai_module = None
                                        try:
                                            # DEBUG: Log Gemini API call details
                                            safe_log_start(
                                                "gemini_api_call_debug",
                                                {
                                                    "gemini_model": gemini_model,
                                                    "api_key_available": bool(api_key),
                                                    "prompt_length": len(entity_prompt),
                                                    "chunk_id": chunk_id
                                                },
                                                project_id
                                            )

                                            # Use pre-fetched gemini_model from outer scope

                                            from google import genai  # type: ignore
                                            genai_module = genai

                                            # DEBUG: Check if Client exists (new API)
                                            if not hasattr(genai, 'Client'):
                                                extraction_logger.log_error(
                                                    Exception("Client not found in google.genai - library may be outdated"),
                                                    "gemini_api_compatibility_error",
                                                    context={
                                                        "available_attrs": [attr for attr in dir(genai) if not attr.startswith('_')],
                                                        "genai_version": getattr(genai, '__version__', 'unknown')
                                                    }
                                                )
                                                return {}

                                            # Use new Google GenAI API (Client-based)
                                            client = genai.Client(api_key=api_key)

                                            response = client.models.generate_content(
                                                model=gemini_model,
                                                contents=entity_prompt,
                                                config={"response_mime_type": "application/json"},
                                            )
                                            response_text = response.text if response else "{}"

                                            # DEBUG: Log response details
                                            safe_log_start(
                                                "gemini_response_debug",
                                                {
                                                    "response_length": len(response_text),
                                                    "response_preview": response_text[:200] if response_text else "EMPTY",
                                                    "chunk_id": chunk_id
                                                },
                                                project_id
                                            )

                                            try:
                                                sanitized_text = _sanitize_json(response_text)
                                                batch_result = json.loads(sanitized_text)

                                                # DEBUG: Log the batch result keys for Gemini
                                                if isinstance(batch_result, dict):
                                                    safe_log_start(
                                                        "gemini_batch_result_debug",
                                                        {
                                                            "result_keys": list(batch_result.keys()),
                                                            "expected_categories": [str(c['id']) for c in category_info if c.get('is_selected')],
                                                            "chunk_id": chunk_id
                                                        },
                                                        project_id
                                                    )
                                                    return batch_result
                                                else:
                                                    extraction_logger.log_warning(
                                                        f"Gemini returned non-dict result: {type(batch_result)}",
                                                        "gemini_invalid_result_type",
                                                        context={"chunk_id": chunk_id, "result_type": str(type(batch_result))}
                                                    )
                                            except Exception as e:
                                                extraction_logger.log_error(
                                                    e,
                                                    "gemini_json_parsing_error",
                                                    context={
                                                        "response_text": response_text[:500],
                                                        "sanitized_text": sanitized_text,
                                                        "chunk_id": chunk_id
                                                    }
                                                )
                                        except ImportError as ie:
                                            extraction_logger.log_error(
                                                ie,
                                                "gemini_import_error",
                                                context={"import_error": str(ie)}
                                            )
                                        except AttributeError as ae:
                                            available_attrs = [attr for attr in dir(genai_module) if not attr.startswith('_')] if genai_module else []
                                            extraction_logger.log_error(
                                                ae,
                                                "gemini_attribute_error",
                                                context={
                                                    "attribute_error": str(ae),
                                                    "available_genai_attrs": available_attrs
                                                }
                                            )
                                        except Exception as e:
                                            extraction_logger.log_error(
                                                e,
                                                "gemini_api_call_error",
                                                context={
                                                    "error_type": type(e).__name__,
                                                    "error_message": str(e),
                                                    "chunk_id": chunk_id
                                                }
                                            )

                                elif effective_classifier == 'ollama':
                                    try:
                                        # Use pre-fetched values from outer scope
                                        api_url = f"{ollama_base_url}/api/generate"
                                        # Prepare Ollama options
                                        options = {
                                            "num_ctx": ollama_num_ctx,
                                            "temperature": ollama_temperature,
                                            "repeat_penalty": ollama_repeat_penalty,
                                            "top_p": ollama_top_p,
                                            "top_k": ollama_top_k,
                                            "num_predict": ollama_num_predict
                                        }
                                        if ollama_seed is not None:
                                            options["seed"] = ollama_seed

                                        # Prepare Ollama request
                                        request_data = {
                                            "model": ollama_model,
                                            "prompt": entity_prompt,
                                            "stream": False,
                                            "raw": True,
                                            "options": options
                                        }

                                        # Add detailed debugging for Ollama calls
                                        safe_log_start(
                                            "ollama_entity_extraction_request",
                                            {
                                                "model": ollama_model,
                                                "api_url": api_url,
                                                "categories_count": len(category_info),
                                                "chunk_id": chunk_id,
                                                "prompt_length": len(entity_prompt),
                                                "request_data_keys": list(request_data.keys())
                                            },
                                            project_id
                                        )

                                        try:
                                            response = requests.post(api_url, json=request_data, timeout=120)
                                        except requests.exceptions.Timeout:
                                            raise
                                        except requests.exceptions.ConnectionError as e:
                                            raise
                                        except Exception as e:
                                            raise

                                        response.raise_for_status()

                                        response_json = response.json()
                                        response_text = response_json.get("response", "{}")

                                        # Log the raw response for debugging
                                        safe_log_start(
                                            "ollama_entity_extraction_response",
                                            {
                                                "response_length": len(response_text),
                                                "response_preview": response_text[:500] if response_text else "EMPTY",
                                                "chunk_id": chunk_id,
                                                "status_code": response.status_code
                                            },
                                            project_id
                                        )

                                        try:
                                            sanitized_text = _sanitize_json(response_text)
                                            result = json.loads(sanitized_text)
                                            if isinstance(result, dict) and "relevant_ids" in result:
                                                relevant_ids = result["relevant_ids"]
                                                return relevant_ids
                                        except Exception as parse_error:
                                            pass

                                        # Try to parse JSON from response
                                        sanitized_text = ""
                                        try:
                                            sanitized_text = _sanitize_json(response_text)
                                            batch_result = json.loads(sanitized_text)
                                            if isinstance(batch_result, dict):
                                                safe_log_start(
                                                    "ollama_entity_extraction_success",
                                                    {
                                                        "parsed_categories": list(batch_result.keys()),
                                                        "chunk_id": chunk_id
                                                    },
                                                    project_id
                                                )
                                                return batch_result
                                            else:
                                                extraction_logger.log_warning(
                                                    f"Ollama returned non-dict result: {type(batch_result)}",
                                                    "ollama_entity_extraction_invalid_format",
                                                    context={"chunk_id": chunk_id, "result_type": str(type(batch_result))}
                                                )
                                        except json.JSONDecodeError as parse_error:
                                            extraction_logger.log_warning(
                                                f"Ollama JSON parsing failed: {parse_error}",
                                                "ollama_entity_extraction_json_error",
                                                context={
                                                    "response_text": response_text[:500],
                                                    "chunk_id": chunk_id,
                                                    "parse_error": str(parse_error)
                                                }
                                            )
                                        except Exception as parse_error:
                                            extraction_logger.log_warning(
                                                f"Ollama response parsing failed: {parse_error}",
                                                "ollama_entity_extraction_parse_error",
                                                context={"chunk_id": chunk_id, "error": str(parse_error)}
                                            )

                                    except requests.exceptions.RequestException as e:
                                        extraction_logger.log_warning(
                                            f"Ollama API request failed: {e}",
                                            "ollama_entity_extraction_api_error",
                                            context={"chunk_id": chunk_id, "error": str(e)}
                                        )
                                    except Exception as e:
                                        extraction_logger.log_warning(
                                            f"Ollama entity extraction failed: {e}",
                                            "ollama_entity_extraction_error",
                                            context={"chunk_id": chunk_id, "error": str(e)}
                                        )

                                elif effective_classifier == 'openai':
                                    api_key = self.openai_api_key
                                    if api_key:
                                        try:
                                            # Use pre-fetched openai_model from outer scope
                                            openai_model = backend_settings.get_classification_openai_model()

                                            import openai  # type: ignore
                                            client = openai.OpenAI(api_key=api_key)

                                            response = client.chat.completions.create(
                                                model=openai_model,
                                                messages=[
                                                    {"role": "system", "content": "You are a precise data extraction assistant. You must output ONLY valid JSON. Do not include any conversational text, markdown formatting, or explanations."},
                                                    {"role": "user", "content": entity_prompt}
                                                ],
                                                response_format={"type": "json_object"},
                                                temperature=0.1
                                            )

                                            response_text = response.choices[0].message.content
                                            if not response_text:
                                                return {}

                                            # Try to parse JSON from response
                                            sanitized_text = ""
                                            try:
                                                sanitized_text = _sanitize_json(response_text)
                                                batch_result = json.loads(sanitized_text)

                                                # DEBUG: Log the batch result keys for OpenAI
                                                if isinstance(batch_result, dict):
                                                    return batch_result
                                                else:
                                                    pass
                                            except Exception as parse_error:
                                                extraction_logger.log_warning(
                                                    f"OpenAI batch JSON parsing failed: {parse_error}",
                                                    "batch_openai_json_parse_error",
                                                    context={
                                                        "response_text": response_text,
                                                        "sanitized_text": sanitized_text if 'sanitized_text' in locals() else "N/A"
                                                    }
                                                )
                                                return {}

                                        except ImportError:
                                            extraction_logger.log_warning(
                                                "OpenAI library not available",
                                                "batch_entity_extraction_missing_dependency"
                                            )
                                        except Exception as e:
                                            extraction_logger.log_warning(
                                                f"OpenAI batch API call failed: {e}",
                                                "batch_entity_extraction_openai_error"
                                            )

                                # For other classifiers, return empty
                                return {}

                            except Exception as e:
                                extraction_logger.log_warning(
                                    f"Batch entity extraction API call failed for {effective_classifier}: {e}",
                                    "batch_entity_extraction_api_call",
                                    context={"categories_count": len(category_info), "chunk_id": chunk_id}
                                )
                                return {}

                        except Exception as e:
                            extraction_logger.log_warning(
                                f"Batch entity extraction failed: {e}",
                                "batch_entity_extraction",
                                context={"chunk_id": chunk_id, "categories_count": len(category_info)}
                            )
                            return {}

                    batch_result = extract_all_entities_with_retry()
                    is_validated = False

                    # Validation stage (optional second call)
                    if enable_validation_stage and batch_result:
                        validation_classifier = validation_classifier or initial_classifier or backend_settings.get_classification_model() or 'grok'
                        
                        # Create validation prompt
                        validation_prompt = f"""You are an expert data validator. Review the following extraction results and validate their accuracy and contextual relevance.

Original extractions:
{json.dumps(batch_result, indent=2)}

Text analyzed:
{chunk_text}

CRITICAL VALIDATION RULES:
1. DO NOT discover new entities or categories. Only work with the ones provided in the 'Original extractions'.
2. VERIFY that every extracted entity/text snippet is actually present in the 'Text analyzed'.
3. REMOVE any extractions that are false positives or contextually irrelevant.
4. ADJUST confidence scores based on how accurately the text reflects the category.
5. If an original extraction is correct, keep it. If it is wrong, remove it.

Return ONLY the validated JSON object.
JSON Response:"""

                        # Perform validation call
                        try:
                            # Log validation start
                            initial_entity_count = sum(len(v.get('entities', v.get('texts', []))) for v in batch_result.values() if isinstance(v, dict))
                            safe_log_start(
                                "validation_stage_started",
                                {
                                    "chunk_id": chunk_id,
                                    "initial_entity_count": initial_entity_count,
                                    "validation_classifier": validation_classifier
                                },
                                project_id
                            )

                            @retry_classification(logger=extraction_logger.logger)
                            def validate_extractions():
                                # Use same API call logic as main extraction
                                if validation_classifier == 'grok' and self.grok_api_key:
                                    api_key = self.grok_api_key
                                    grok_model = backend_settings.get_classification_grok_model()

                                    api_url = "https://api.x.ai/v1/chat/completions"
                                    request_data = {
                                        "model": grok_model,
                                        "messages": [
                                            {"role": "system", "content": "You are a precise data validation assistant. You must output ONLY valid JSON. Do not add new data, only filter and refine existing data."},
                                            {"role": "user", "content": validation_prompt}
                                        ],
                                        "temperature": 0.1
                                    }
                                    headers = {
                                        "Authorization": f"Bearer {api_key}",
                                        "Content-Type": "application/json"
                                    }

                                    response = requests.post(api_url, json=request_data, headers=headers, timeout=180)
                                    response.raise_for_status()

                                    response_json = response.json()
                                    response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "{}")

                                    try:
                                        from ...features.extraction.grok_classifier import _sanitize_json
                                        sanitized_text = _sanitize_json(response_text)
                                        validated_result = json.loads(sanitized_text)
                                        if isinstance(validated_result, dict):
                                            return validated_result
                                    except Exception:
                                        pass

                                # Add similar logic for gemini and ollama if needed
                                return batch_result  # Fall back to original if validation fails

                            validated_batch = validate_extractions()
                            
                            # Programmatic Deduplication and Filtering
                            if isinstance(validated_batch, dict) and validated_batch:
                                # Ensure we don't allow "discovery" of new categories during validation
                                discovered_categories = [k for k in validated_batch.keys() if k not in batch_result]
                                if discovered_categories:
                                    safe_log_warning(
                                        f"Validation stage tried to discover {len(discovered_categories)} new categories - dropping them.",
                                        "validation_discovery_prevention",
                                        {"discovered_categories": discovered_categories},
                                        project_id
                                    )

                                filtered_batch = {k: v for k, v in validated_batch.items() if k in batch_result}
                                
                                # Log validation results
                                validated_entity_count = sum(len(v.get('entities', v.get('texts', []))) for v in filtered_batch.values() if isinstance(v, dict))
                                safe_log_start(
                                    "validation_stage_complete",
                                    {
                                        "chunk_id": chunk_id,
                                        "initial_count": initial_entity_count,
                                        "validated_count": validated_entity_count,
                                        "removed_count": initial_entity_count - validated_entity_count
                                    },
                                    project_id
                                )
                                
                                batch_result = filtered_batch
                                is_validated = True

                        except Exception as e:
                            extraction_logger.log_warning(
                                f"Validation stage failed: {e}",
                                "validation_stage_error",
                                context={"chunk_id": chunk_id, "validation_classifier": validation_classifier}
                            )
                            # Continue with original results if validation fails

                    # Process the batch results for each selected category only
                    for cat_info in category_info:
                        cat_id = str(cat_info['id'])
                        category_name = cat_info['name']
                        is_selected = cat_info.get('is_selected', False)

                        # Only extract entities for selected categories, not parent categories
                        if not is_selected:
                            continue

                        # Extract results for this selected category from the batch response (trying both ID types)
                        category_result = batch_result.get(cat_id) or batch_result.get(cat_info['id'], {})

                        if extraction_type == "whole_text":
                            # Handle whole text extraction results
                            texts_found = category_result.get('texts', [])
                            confidence = category_result.get('confidence', 0.0)

                            # Filter out empty or invalid texts
                            valid_texts = [str(text).strip() for text in texts_found
                                          if text and str(text).strip() and len(str(text).strip()) > 10]  # Minimum length for meaningful text

                            if valid_texts:
                                # Deduplicate texts
                                unique_texts = list(dict.fromkeys(valid_texts))
                                
                                extracted_entities[cat_id] = {
                                    'category_name': category_name,
                                    'texts': unique_texts,
                                    'confidence': confidence
                                }
                                entity_confidences.update({f"text_{i}": confidence for i, _ in enumerate(unique_texts)})
                        else:
                            # Handle NER extraction results (existing logic)
                            entities_found = category_result.get('entities', [])
                            confidence = category_result.get('confidence', 0.0)

                            # Filter out empty or invalid entities
                            valid_entities = [str(entity).strip() for entity in entities_found
                                            if entity and str(entity).strip() and len(str(entity).strip()) > 1]

                            if valid_entities:
                                # Deduplicate entities
                                unique_entities = list(dict.fromkeys(valid_entities))
                                
                                extracted_entities[cat_id] = {
                                    'category_name': category_name,
                                    'entities': unique_entities,
                                    'confidence': confidence
                                }
                                entity_confidences.update({entity: confidence for entity in unique_entities})

                            # Skip verbose per-category success logging

                    # Only include chunks that have extracted entities for at least one category
                    if not extracted_entities:
                        continue
                    
                    # Note: Relationship inference has been removed from this implementation
                    relationships = []

                    # Calculate overall confidence for this chunk
                    overall_confidence = max(entity_confidences.values()) if entity_confidences else 0.0

                    # Create extraction result with extraction data
                    result = ExtractionResult(
                        job_id=0,  # TODO: Pass actual job_id when available
                        project_id=project_id,
                        chunk_id=str(chunk_id),
                        categories=list(extracted_entities.keys()),
                        confidence=overall_confidence,
                        extracted_data={
                            'is_validated': is_validated,
                            'processing_time': time.time() - chunk_start_time,
                            'extracted_entities': extracted_entities,
                            'entity_confidences': entity_confidences,
                            'extraction_type': extraction_type
                        }
                    )
                    results.append(result)
                    processed_count += 1
                except Exception as e:
                    failed_chunks += 1
                    chunk_id = chunk_data.get('id', 'unknown') if chunk_data else 'unknown'
                    
                    pid_int = None
                    if isinstance(project_id, int):
                        pid_int = project_id
                    elif isinstance(project_id, str) and project_id.isdigit():
                        pid_int = int(project_id)

                    extraction_logger.log_error(
                        e,
                        "chunk_processing",
                        context={
                            "chunk_id": chunk_id,
                            "chunk_length": len(chunk_data.get('text', '')) if chunk_data else 0,
                            "processing_time": time.time() - chunk_start_time
                        },
                        project_id=pid_int
                    )
                    continue

            # Log performance metrics
            total_time = time.time() - start_time
            if processed_count > 0:
                avg_chunk_time = total_time / len(chunks_data)
                
                pid_int = None
                if isinstance(project_id, int):
                    pid_int = project_id
                elif isinstance(project_id, str) and project_id.isdigit():
                    pid_int = int(project_id)

                extraction_logger.log_operation_complete(
                    "extract_content",
                    total_time,
                    result_count=processed_count,
                    context={
                        "chunks_processed": processed_count,
                        "chunks_failed": failed_chunks,
                        "avg_chunk_time": avg_chunk_time,
                        "success_rate": processed_count / len(chunks_data) if chunks_data else 0
                    },
                    project_id=pid_int
                )
            else:
                 pass

            # Create summary
            summary = ExtractionSummary(
                total_chunks=len(chunks_data),
                processed_chunks=processed_count,
                filtered_chunks=len(chunks_data) - processed_count - failed_chunks,
                categories_used=list(target_category_ids) if target_category_ids else [],
                confidence_threshold=confidence_threshold,
                extraction_time=total_time
            )

            return results, summary

        except (TaxonomyNotFoundError, TaxonomyValidationError, TaxonomyCategoryError,
                ClassificationFailureError, ChunkProcessingError):
            # Re-raise domain-specific exceptions
            raise
        except Exception as e:
            pid_int = None
            if isinstance(project_id, int):
                pid_int = project_id
            elif isinstance(project_id, str) and project_id.isdigit():
                pid_int = int(project_id)

            extraction_logger.log_error(
                e,
                "extract_content",
                context={
                    "project_id": project_id,
                    "taxonomy_project": taxonomy_project,
                    "taxonomy_name": taxonomy_name,
                    "duration": time.time() - start_time
                },
                project_id=pid_int
            )
            raise ClassificationFailureError(f"Extraction failed: {e}") from e


    def _resolve_category_paths(self, taxonomy: HierarchicalCategory, category_paths: List[str]) -> List[HierarchicalCategory]:
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
        resolved_categories = self.category_resolver.resolve_category_paths(taxonomy, category_paths)

        # If no categories were resolved and we had multiple paths, raise an error
        if not resolved_categories and len(category_paths) > 1:
            raise TaxonomyCategoryError(f"All category paths failed to resolve: {category_paths}", str(category_paths))

        return resolved_categories

    def _check_category_match(self, classifications: Dict[str, Any], target_categories: List[HierarchicalCategory]) -> List[str]:
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
        return self.category_resolver.check_category_match(classifications, target_categories)

    def _load_project_chunks(self, project_id: int, max_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load chunks for a project from processed outputs.

        Args:
            project_id: Project ID
            max_chunks: Maximum number of chunks to load

        Returns:
            List of chunk data dictionaries

        Raises:
            DatabaseConnectionError: If database query fails
            FileStorageError: If file reading fails
        """
        return self.chunk_loader.load_project_chunks(project_id, max_chunks)