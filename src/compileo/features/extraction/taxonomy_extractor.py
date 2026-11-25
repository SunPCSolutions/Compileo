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
from src.compileo.core.settings import backend_settings, BackendSettings
from src.compileo.core.openai_compat import openai_sampling_options
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
    """
    Service for selective content extraction using taxonomy-based classification.
    """

    def __init__(self,
                 taxonomy_repo: TaxonomyRepository,
                 chunk_repo: ChunkRepository,
                 file_manager: FileManager,
                 processed_output_repo=None,  # For backward compatibility with extraction result storage
                 grok_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 ollama_available: bool = True):
        """
        Initialize TaxonomyExtractor.
        """
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

            # Database connection is managed per-request in multi-tenant mode
            # Repositories now use owner_id to get connections on demand

            from src.compileo.storage.src.project.database_repositories import ProjectRepository
            # Get owner_id from taxonomy_repo for ProjectRepository
            owner_id = getattr(taxonomy_repo, 'owner_id', None)
            self.user_settings = BackendSettings(owner_id=owner_id)
            
            self.taxonomy_loader = TaxonomyLoader(taxonomy_repo, ProjectRepository(owner_id))
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
                for parent_name in category.parent_path:
                    for cat_id_in_map, cat_obj in category_map.items():
                        if cat_obj.name == parent_name and cat_id_in_map not in expanded_ids:
                            expanded_ids.add(cat_id_in_map)
                            break

        return expanded_ids


    def _traverse_taxonomy(self, taxonomy: HierarchicalCategory):
        """
        Traverse the taxonomy tree and yield all nodes.
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
        """
        Extract content from chunks using taxonomy-based classification with selective filtering.
        """
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
                    taxonomy = self.taxonomy_loader.load_taxonomy_by_project(taxonomy_project, "")
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

                    # Log progress periodically
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
                                    
                                    # LOGGING
                                    if node_id_str and node_id_str in target_category_ids:
                                        extraction_logger.log_warning(
                                            f"Category {cat_id} ('{current_category.name}') has a parent '{node.name}' (ID: {node_id_str}) that is also in target_category_ids.",
                                            "parent_context_debug",
                                            context={"child_id": cat_id, "parent_id": node_id_str, "in_targets": True},
                                            project_id=int(project_id) if isinstance(project_id, (str, int)) and str(project_id).isdigit() else None
                                        )

                                    if node_id_str:
                                        parent_cat_info = self.category_resolver.get_category_info_by_id(taxonomy, str(node.id) if hasattr(node, 'id') else "")
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
                            # Root level category
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

                            if extraction_mode == "contextual":
                                current_category = None
                                for node in self._traverse_taxonomy(taxonomy):
                                    if hasattr(node, 'id') and node.id == cat_id:
                                        current_category = node
                                        break

                                if current_category and hasattr(current_category, 'parent_path') and current_category.parent_path:
                                    immediate_parent_name = current_category.parent_path[-1]
                                    for node in self._traverse_taxonomy(taxonomy):
                                        if (hasattr(node, 'name') and node.name == immediate_parent_name):
                                            parent_cat_info = self.category_resolver.get_category_info_by_id(taxonomy, str(node.id) if hasattr(node, 'id') else "")
                                            if parent_cat_info and parent_cat_info.get('name'):
                                                category_info.append({
                                                    'id': str(node.id) if hasattr(node, 'id') else "",
                                                    'name': parent_cat_info['name'],
                                                    'is_selected': False,
                                                    'is_parent': True
                                                })
                                                break

                    if not category_info:
                        continue

                    # Create parent-child structured prompt
                    parent_child_text = ""
                    if extraction_mode == "contextual":
                        parent_child_text = "CONTEXTUAL HIERARCHY (Extract only if child matches parent context):\n"
                        for parent_id, parent_data in parent_child_structure.items():
                            parent_child_text += f"\nParent Context: {parent_data['name']} (ID: {parent_id})\n"
                            if parent_data.get('description'):
                                parent_child_text += f"  Context Description: {parent_data['description']}\n"
                            parent_child_text += f"  Target Categories under this context:\n"
                            for child in parent_data['children']:
                                parent_child_text += f"    - {child['name']} (ID: {child['id']})\n"
                                if child.get('description'):
                                    parent_child_text += f"      Category Description: {child['description']}\n"
                    else:
                        parent_child_text = "Selected categories to extract from (independent of parent relationships):\n"
                        for cat_info in category_info:
                            if cat_info.get('is_selected', False):
                                parent_child_text += f"ID: {cat_info['id']} - Name: {cat_info['name']}\n"
                                if cat_info.get('description'):
                                    parent_child_text += f"      Description: {cat_info['description']}\n"

                    effective_classifier = initial_classifier or self.user_settings.get_classification_model() or 'grok'
                    entity_prompt = ""

                    if extraction_type == "whole_text":
                        if extraction_mode == "document_wide":
                            # Document-wide prompt
                            entity_prompt = f"""Extract complete text portions from the following content for each of the specified categories.

{parent_child_text}

CRITICAL DOCUMENT-WIDE RULES:
- Process ALL specified categories regardless of content context or topic relevance.
- Do not filter or skip categories based on whether their parent topics are discussed.
- Extract text portions for every category in the list, even if the text doesn't seem related.
- **CLEAN TEXT**: Remove all formatting artifacts like bullet points (*, -, •), numbering (1., 2.), and extra whitespace. Return only the clean sentence/paragraph.
- If no text portions are found for a category, return an empty array (don't skip the category).
- YOU MUST USE THE EXACT CATEGORY ID AS THE KEY IN THE JSON RESPONSE. DO NOT USE THE CATEGORY NAME.
- DO NOT INCLUDE PARENT IDs IN THE JSON RESPONSE UNLESS THEY ARE ALSO LISTED AS TARGET CATEGORIES.

Return your response as a JSON object where each key is the exact category ID provided above, and each value is an object with:
- "texts": array of strings (the CLEANED text portions found for this category)
- "confidence": number between 0.0 and 1.0 (REQUIRED field, e.g. 0.95). DO NOT use strings like "High" or "Low".

Text to analyze:
{chunk_text}

JSON Response:"""
                        else:
                            # Contextual prompt
                            entity_prompt = f"""You are a meticulous Data Categorization Auditor. Your goal is to extract text segments that strictly match a specific hierarchical context.

{parent_child_text}

**YOUR THOUGHT PROCESS:**
For each target category ID, you must iterate through the text and perform the following check for every potential match:
1.  **Identify Content**: Does this sentence discuss the target category?
2.  **Identify Context**: Does this sentence discuss it specifically in the context of its ASSIGNED PARENT?
3.  **Strict Filtering**: Only if BOTH the category AND the parent context match the text exactly should you extract it.

**CRITICAL RULES:**
- **Zero Tolerance for Context Mismatch**: If a text segment belongs to the target category concept but falls under a different parent domain, you MUST exclude it. Return an empty array [] for that ID.
- **Explicit Evidence**: Do not infer context. The text must explicitly or strongly imply the specific parent context.
- **Content over Headers**: Focus on the *content* of the text, not just section headers. Relevant information may appear in sections with different titles.
- **CLEAN TEXT**: Remove all formatting artifacts like bullet points (*, -, •), numbering (1., 2.), and extra whitespace. Return only the clean sentence/paragraph.
- **JSON Keys**: Use ONLY the provided category IDs. Do not use names.

**Output Format:**
Return a JSON object with a single key "results" containing a list of extraction objects.
Example:
{{
  "results": [
    {{
      "category_id": "110100",
      "texts": ["extracted text 1", "extracted text 2"],
      "confidence": 0.95
    }}
  ]
}}

Text to analyze:
{chunk_text}

JSON Response:"""
                    else:
                        # NER extraction prompts
                        if extraction_mode == "document_wide":
                            entity_prompt = f"""Extract entities from the following content for each of the specified categories.

{parent_child_text}

CRITICAL DOCUMENT-WIDE RULES:
- Process ALL specified categories regardless of content context or topic relevance
- Do not filter or skip categories based on whether their parent topics are discussed
- Extract entities for every category in the list, even if the text doesn't seem related
- **CLEAN ENTITIES**: Remove bullet points (*, -), numbering, and extra whitespace from extracted entities.
- If no entities are found for a category, return an empty array (don't skip the category)
- YOU MUST USE THE EXACT CATEGORY ID AS THE KEY IN THE JSON RESPONSE. DO NOT USE THE CATEGORY NAME.

Return your response as a JSON object where each key is the exact category ID provided above, and each value is an object with:
- "entities": array of strings (the CLEANED entities found for this category)
- "confidence": floating point number between 0.0 and 1.0 representing certainty (e.g. 0.95). MUST BE A NUMBER.

Text to analyze:
{chunk_text}

JSON Response:"""
                        else:
                            # Contextual NER
                            entity_prompt = f"""You are a meticulous Data Categorization Auditor. Your goal is to extract specific entities that strictly match a hierarchical context.

{parent_child_text}

**YOUR THOUGHT PROCESS:**
For each target category ID, analyze potential entities:
1.  **Entity Match**: Is this entity an instance of the target category?
2.  **Context Match**: Is this entity mentioned in the context of the ASSIGNED PARENT?
3.  **Strict Filtering**: REJECT entities if they appear in a different context.

**CRITICAL RULES:**
- **Zero Tolerance**: If the context doesn't match the assigned parent, return an empty array [] for that ID.
- **Content over Headers**: Focus on the *content* of the text, not just section headers. Relevant information may appear in sections with different titles.
- **CLEAN ENTITIES**: Remove bullet points (*, -), numbering, and extra whitespace.
- **JSON Keys**: Use ONLY the provided category IDs.

**Output Format:**
Return a JSON object with a single key "results" containing a list of extraction objects.
Example:
{{
  "results": [
    {{
      "category_id": "110100",
      "entities": ["entity 1", "entity 2"],
      "confidence": 0.95
    }}
  ]
}}

Text to analyze:
{chunk_text}

JSON Response:"""

                    # Extract entities for all categories in a single API call
                    grok_model = self.user_settings.get_classification_grok_model() if effective_classifier == 'grok' else None
                    gemini_model = self.user_settings.get_classification_gemini_model() if effective_classifier == 'gemini' else None
                    openai_model = self.user_settings.get_classification_openai_model() if effective_classifier == 'openai' else None
                    ollama_model = self.user_settings.get_classification_ollama_model() if effective_classifier == 'ollama' else None
                    ollama_base_url = self.user_settings.get_ollama_base_url() if effective_classifier == 'ollama' else None
                    ollama_num_ctx = self.user_settings.get_classification_ollama_num_ctx() if effective_classifier == 'ollama' else None
                    ollama_temperature = self.user_settings.get_classification_ollama_temperature() if effective_classifier == 'ollama' else None
                    ollama_repeat_penalty = self.user_settings.get_classification_ollama_repeat_penalty() if effective_classifier == 'ollama' else None
                    ollama_top_p = self.user_settings.get_classification_ollama_top_p() if effective_classifier == 'ollama' else None
                    ollama_top_k = self.user_settings.get_classification_ollama_top_k() if effective_classifier == 'ollama' else None
                    ollama_num_predict = self.user_settings.get_classification_ollama_num_predict() if effective_classifier == 'ollama' else None
                    ollama_seed = self.user_settings.get_classification_ollama_seed() if effective_classifier == 'ollama' else None

                    @retry_classification(logger=extraction_logger.logger)
                    def extract_all_entities_with_retry():
                        try:
                            try:
                                from google import genai # type: ignore
                            except ImportError:
                                pass

                            try:
                                from ...features.extraction.grok_classifier import _sanitize_json

                                if effective_classifier == 'grok':
                                    api_key = self.grok_api_key or ""
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

                                    response = requests.post(api_url, json=request_data, headers=headers, timeout=180)
                                    response.raise_for_status()

                                    response_json = response.json()
                                    response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content") or "{}"

                                    sanitized_text = ""
                                    try:
                                        sanitized_text = _sanitize_json(response_text)
                                        parsed_json = json.loads(sanitized_text)
                                        
                                        # Transform "results" list format back to dictionary format: {cat_id: {texts/entities: [], confidence: float}}
                                        if isinstance(parsed_json, dict) and "results" in parsed_json and isinstance(parsed_json["results"], list):
                                            batch_result = {}
                                            for item in parsed_json["results"]:
                                                cat_id = item.get("category_id")
                                                if cat_id:
                                                    batch_result[cat_id] = item
                                            return batch_result
                                        
                                        # Fallback for legacy format
                                        if isinstance(parsed_json, dict):
                                            return parsed_json
                                            
                                    except Exception as parse_error:
                                        extraction_logger.log_error(
                                            parse_error,
                                            "batch_grok_json_parse_debug",
                                            context={"response_text_length": len(response_text)}
                                        )
                                        return {}

                                elif effective_classifier == 'gemini':
                                    api_key = self.gemini_api_key
                                    if api_key:
                                        sanitized_text = "N/A"
                                        try:
                                            from google import genai  # type: ignore
                                            client = genai.Client(api_key=api_key)

                                            response = client.models.generate_content(
                                                model=gemini_model,
                                                contents=entity_prompt,
                                                config={"response_mime_type": "application/json"},
                                            )
                                            response_text = response.text if response and hasattr(response, 'text') and response.text else "{}"

                                            try:
                                                sanitized_text = _sanitize_json(response_text)
                                                parsed_json = json.loads(sanitized_text)
                                                
                                                # Transform "results" list format back to dictionary format
                                                if isinstance(parsed_json, dict) and "results" in parsed_json and isinstance(parsed_json["results"], list):
                                                    batch_result = {}
                                                    for item in parsed_json["results"]:
                                                        cat_id = item.get("category_id")
                                                        if cat_id:
                                                            batch_result[cat_id] = item
                                                    return batch_result

                                                # Fallback for legacy format
                                                if isinstance(parsed_json, dict):
                                                    return parsed_json
                                            except Exception as e:
                                                extraction_logger.log_error(e, "gemini_json_parsing_error")
                                        except Exception as e:
                                            extraction_logger.log_error(e, "gemini_api_call_error")

                                elif effective_classifier == 'ollama':
                                    try:
                                        api_url = f"{ollama_base_url}/api/generate"
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

                                        request_data = {
                                            "model": ollama_model,
                                            "prompt": entity_prompt,
                                            "stream": False,
                                            "raw": True,
                                            "options": options
                                        }

                                        response = requests.post(api_url, json=request_data, timeout=120)
                                        response.raise_for_status()

                                        response_json = response.json()
                                        response_text = response_json.get("response") or "{}"

                                        try:
                                            sanitized_text = _sanitize_json(response_text)
                                            parsed_json = json.loads(sanitized_text)
                                            
                                            # Transform "results" list format back to dictionary format
                                            if isinstance(parsed_json, dict) and "results" in parsed_json and isinstance(parsed_json["results"], list):
                                                batch_result = {}
                                                for item in parsed_json["results"]:
                                                    cat_id = item.get("category_id")
                                                    if cat_id:
                                                        batch_result[cat_id] = item
                                                return batch_result

                                            # Fallback for legacy format
                                            if isinstance(parsed_json, dict):
                                                return parsed_json
                                        except Exception:
                                            pass

                                    except Exception as e:
                                        extraction_logger.log_warning(f"Ollama entity extraction failed: {e}", "ollama_error")

                                elif effective_classifier == 'openai':
                                    api_key = self.openai_api_key
                                    if api_key:
                                        try:
                                            import openai  # type: ignore
                                            client = openai.OpenAI(api_key=api_key)

                                            response = client.chat.completions.create(
                                                model=openai_model,
                                                messages=[
                                                    {"role": "system", "content": "You are a precise data extraction assistant. You must output ONLY valid JSON."},
                                                    {"role": "user", "content": entity_prompt}
                                                ],
                                                response_format={"type": "json_object"},
                                                **openai_sampling_options(openai_model, temperature=0.1),
                                            )

                                            response_text = response.choices[0].message.content
                                            if response_text:
                                                sanitized_text = _sanitize_json(response_text)
                                                parsed_json = json.loads(sanitized_text)
                                                
                                                # Transform "results" list format back to dictionary format
                                                if isinstance(parsed_json, dict) and "results" in parsed_json and isinstance(parsed_json["results"], list):
                                                    batch_result = {}
                                                    for item in parsed_json["results"]:
                                                        cat_id = item.get("category_id")
                                                        if cat_id:
                                                            batch_result[cat_id] = item
                                                    return batch_result

                                                return parsed_json
                                        except Exception:
                                            pass

                                return {}

                            except Exception:
                                return {}

                        except Exception:
                            return {}

                    batch_result = extract_all_entities_with_retry()
                    
                    # Robust check for batch_result - must be a dictionary
                    if not isinstance(batch_result, dict):
                        extraction_logger.log_warning(
                            f"AI classifier returned {type(batch_result).__name__} instead of dict. Normalizing to empty dict.",
                            "normalization",
                            context={"chunk_id": str(chunk_id), "type": type(batch_result).__name__}
                        )
                        batch_result = {}
                        
                    is_validated = False

                    if enable_validation_stage and batch_result:
                        validation_classifier = validation_classifier or initial_classifier or self.user_settings.get_classification_model() or 'grok'
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

                        try:
                            @retry_classification(logger=extraction_logger.logger)
                            def validate_extractions():
                                if validation_classifier == 'grok' and self.grok_api_key:
                                    api_key = self.grok_api_key
                                    grok_model = self.user_settings.get_classification_grok_model()

                                    api_url = "https://api.x.ai/v1/chat/completions"
                                    request_data = {
                                        "model": grok_model,
                                        "messages": [
                                            {"role": "system", "content": "You are a precise data validation assistant. You must output ONLY valid JSON."},
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
                                        return json.loads(sanitized_text)
                                    except Exception:
                                        pass
                                return batch_result

                            validated_batch = validate_extractions()
                            if isinstance(validated_batch, dict) and validated_batch:
                                filtered_batch = {k: v for k, v in validated_batch.items() if k in batch_result}
                                batch_result = filtered_batch
                                is_validated = True

                        except Exception:
                            pass

                    # Process results
                    extracted_entities = {}
                    entity_confidences = {}

                    for cat_info in category_info:
                        cat_id = str(cat_info['id'])
                        category_name = cat_info['name']
                        is_selected = cat_info.get('is_selected', False)

                        if not is_selected:
                            continue

                        # batch_result is guaranteed to be a dict here due to normalization above
                        category_result = batch_result.get(cat_id, {})

                        # Robust normalization for AI output variations
                        if isinstance(category_result, list):
                            # AI returned a list directly instead of an object with 'texts'/'entities' key.
                            # We use 0.0 as fallback confidence because the AI failed to provide an assessment.
                            if extraction_type == "whole_text":
                                category_result = {'texts': category_result, 'confidence': 0.0}
                            else:
                                category_result = {'entities': category_result, 'confidence': 0.0}
                        elif not isinstance(category_result, dict):
                            category_result = {}

                        if extraction_type == "whole_text":
                            texts_found = category_result.get('texts', [])
                            confidence_val = category_result.get('confidence', 0.0)
                            try:
                                confidence = float(confidence_val)
                            except (ValueError, TypeError):
                                confidence = 0.0
                            
                            valid_texts = []
                            for text in texts_found:
                                if not text:
                                    continue
                                
                                text_val = ""
                                
                                # Handle case where AI returns an object with a 'text' field instead of a string
                                if isinstance(text, dict):
                                    text_val = str(text.get('text', text)).strip()
                                    # Attempt to recover confidence from item if overall confidence was missing
                                    if confidence == 0.0:
                                        try:
                                            item_conf = text.get('confidence')
                                            if item_conf is not None:
                                                confidence = float(item_conf)
                                        except (ValueError, TypeError):
                                            pass
                                elif isinstance(text, str):
                                    # Try to parse stringified dict if it looks like one
                                    text = text.strip()
                                    if text.startswith("{") and "confidence" in text:
                                        try:
                                            # Try simple replace for common python dict string format
                                            if "'" in text:
                                                import ast
                                                parsed = ast.literal_eval(text)
                                                if isinstance(parsed, dict):
                                                    text_val = str(parsed.get('text', text)).strip()
                                                    if confidence == 0.0:
                                                        item_conf = parsed.get('confidence')
                                                        if item_conf is not None:
                                                            confidence = float(item_conf)
                                            else:
                                                parsed = json.loads(text)
                                                text_val = str(parsed.get('text', text)).strip()
                                                if confidence == 0.0:
                                                    item_conf = parsed.get('confidence')
                                                    if item_conf is not None:
                                                        confidence = float(item_conf)
                                        except:
                                            text_val = text
                                    else:
                                        text_val = text
                                else:
                                    text_val = str(text).strip()
                                
                                # Clean text formatting (remove bullets, etc.)
                                if text_val:
                                    import re
                                    # Remove leading bullets (*, -, •, +)
                                    text_val = re.sub(r'^[\*\-\•\+]\s*', '', text_val)
                                    # Remove leading numbering (1., 1.1)
                                    text_val = re.sub(r'^\d+(\.\d+)*\.?\s*', '', text_val)
                                    # Remove leading "CategoryName:" patterns
                                    text_val = re.sub(r'^[^:]+:\s*', '', text_val)
                                    text_val = text_val.strip()

                                if len(text_val) > 10:
                                    valid_texts.append(text_val)

                            if valid_texts:
                                unique_texts = list(dict.fromkeys(valid_texts))
                                extracted_entities[cat_id] = {
                                    'category_name': category_name,
                                    'texts': unique_texts,
                                    'confidence': confidence
                                }
                                entity_confidences.update({f"text_{i}": confidence for i, _ in enumerate(unique_texts)})
                        else:
                            entities_found = category_result.get('entities', [])
                            confidence_val = category_result.get('confidence', 0.0)
                            try:
                                confidence = float(confidence_val)
                            except (ValueError, TypeError):
                                confidence = 0.0
                            
                            valid_entities = []
                            for entity in entities_found:
                                if not entity:
                                    continue
                                    
                                # Handle case where AI returns an object with an 'entity' field instead of a string
                                if isinstance(entity, dict) and 'entity' in entity:
                                    entity_val = str(entity['entity']).strip()
                                elif isinstance(entity, dict) and 'name' in entity:
                                    entity_val = str(entity['name']).strip()
                                else:
                                    entity_val = str(entity).strip()
                                
                                # Clean text formatting
                                if entity_val:
                                    import re
                                    # Remove leading bullets (*, -, •, +)
                                    entity_val = re.sub(r'^[\*\-\•\+]\s*', '', entity_val)
                                    # Remove leading numbering (1., 1.1)
                                    entity_val = re.sub(r'^\d+(\.\d+)*\.?\s*', '', entity_val)
                                    # Remove leading "CategoryName:" patterns
                                    entity_val = re.sub(r'^[^:]+:\s*', '', entity_val)
                                    entity_val = entity_val.strip()

                                if len(entity_val) > 1:
                                    valid_entities.append(entity_val)

                            if valid_entities:
                                unique_entities = list(dict.fromkeys(valid_entities))
                                extracted_entities[cat_id] = {
                                    'category_name': category_name,
                                    'entities': unique_entities,
                                    'confidence': confidence
                                }
                                entity_confidences.update({entity: confidence for entity in unique_entities})

                    if not extracted_entities:
                        continue
                    
                    overall_confidence = max(entity_confidences.values()) if entity_confidences else 0.0
                    result = ExtractionResult(
                        job_id=0,
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
                    extraction_logger.log_error(e, "chunk_processing")
                    continue

            total_time = time.time() - start_time
            summary = ExtractionSummary(
                total_chunks=len(chunks_data),
                processed_chunks=processed_count,
                filtered_chunks=len(chunks_data) - processed_count - failed_chunks,
                categories_used=list(target_category_ids) if target_category_ids else [],
                confidence_threshold=confidence_threshold,
                extraction_time=total_time
            )

            return results, summary

        except Exception as e:
            extraction_logger.log_error(e, "extract_content")
            raise ClassificationFailureError(f"Extraction failed: {e}") from e


    def _resolve_category_paths(self, taxonomy: HierarchicalCategory, category_paths: List[str]) -> List[HierarchicalCategory]:
        """
        Resolve category path strings to actual category objects.
        """
        resolved_categories = self.category_resolver.resolve_category_paths(taxonomy, category_paths)
        if not resolved_categories and len(category_paths) > 1:
            raise TaxonomyCategoryError(f"All category paths failed to resolve: {category_paths}", str(category_paths))
        return resolved_categories

    def _check_category_match(self, classifications: Dict[str, Any], target_categories: List[HierarchicalCategory]) -> List[str]:
        """
        Check if classifications match any of the target categories.
        """
        return self.category_resolver.check_category_match(classifications, target_categories)

    def _load_project_chunks(self, project_id: int, max_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load chunks for a project from processed outputs.
        """
        return self.chunk_loader.load_project_chunks(project_id, max_chunks)
