import logging
from typing import Optional, Dict, Any, Union, List
from src.compileo.storage.src.project.database_repositories import PromptRepository, DatasetParameterRepository
from .constants import (
    QA_RESPONSE_SCHEMA, QUESTION_GENERATION_SCHEMA, ANSWER_GENERATION_SCHEMA,
    SUMMARIZATION_SCHEMA, build_json_instructions, DEFAULT_DATASET_PARAMS
)


class PromptBuilder:
    """Builds prompts by combining templates and dataset parameters."""
    
    def __init__(self, prompt_repository: PromptRepository, 
                 dataset_parameter_repository: DatasetParameterRepository):
        self.prompt_repository = prompt_repository
        self.dataset_parameter_repository = dataset_parameter_repository

    def _safe_format(self, template: str, param_dict: dict) -> str:
        """Safely formats a template by replacing placeholders."""
        result = template
        for key, value in param_dict.items():
            placeholder = "{" + str(key) + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    def _get_prompt_template(self, prompt_name: str, project_id: str, owner_id: str) -> str:
        """Fetch and extract prompt template from repository."""
        prompt_record = self.prompt_repository.get_by_name(prompt_name, str(project_id), owner_id)
        if not prompt_record:
            raise ValueError(f"Prompt '{prompt_name}' not found.")

        if isinstance(prompt_record, dict):
            return prompt_record.get('content', '')
        elif isinstance(prompt_record, (tuple, list)):
            return prompt_record[2]
        return str(prompt_record)

    def _build_base_prompt(self, template: str, chunk_text: Optional[str], 
                          param_dict: dict) -> str:
        """Build base prompt with chunk text insertion."""
        if chunk_text is not None:
            if "{chunk}" in template:
                template = template.replace("{chunk}", chunk_text)
            else:
                template = f"Content: {chunk_text}\n\n{template}"
        return self._safe_format(template, param_dict)

    def build_prompt(self, project_id: str, owner_id: str, prompt_name: str,
                    chunk_text: Optional[str] = None,
                    taxonomy_project: Optional[str] = None,
                    taxonomy_name: Optional[str] = None,
                    datasets_per_chunk: int = 3) -> str:
        """Build main QA generation prompt.
        
        Args:
            datasets_per_chunk: Number of entries per chunk. Use 0 for AI-determined count.
        """
        prompt_template = self._get_prompt_template(prompt_name, project_id, owner_id)
        param_dict = self.get_param_dict(project_id, owner_id)
        param_dict['datasets_per_chunk'] = datasets_per_chunk if datasets_per_chunk > 0 else "AI-determined"

        # Load taxonomy info if specified
        taxonomy_info = self._load_taxonomy_info(taxonomy_project, taxonomy_name, owner_id)
        if taxonomy_info:
            param_dict.update(taxonomy_info)

        final_prompt = self._build_base_prompt(prompt_template, chunk_text, param_dict)

        # Add QA-specific instructions
        if "question" in final_prompt.lower() or "answer" in final_prompt.lower():
            # Use AI-determined schema and instructions when datasets_per_chunk is 0
            if datasets_per_chunk == 0:
                from .constants import QA_RESPONSE_ARRAY_SCHEMA
                final_prompt += build_json_instructions(
                    QA_RESPONSE_ARRAY_SCHEMA,
                    ai_decides=True
                )
            else:
                final_prompt += build_json_instructions(
                    QA_RESPONSE_SCHEMA,
                    multiple_items=datasets_per_chunk
                )

        return final_prompt

    def _load_taxonomy_info(self, taxonomy_project: Optional[str], 
                           taxonomy_name: Optional[str],
                           owner_id: str) -> Optional[Dict[str, str]]:
        """Load taxonomy information if specified."""
        if not taxonomy_project or not taxonomy_name:
            return None
            
        from src.compileo.features.taxonomy.loader import TaxonomyLoader
        from src.compileo.storage.src.project.database_repositories import TaxonomyRepository
        from src.compileo.storage.src.database import get_db_connection
        
        try:
            from src.compileo.storage.src.project.database_repositories import ProjectRepository
            taxonomy_repo = TaxonomyRepository(owner_id)
            project_repo = ProjectRepository(owner_id)
            taxonomy_loader = TaxonomyLoader(taxonomy_repo, project_repo)
            taxonomy = taxonomy_loader.load_taxonomy_by_project(
                taxonomy_project, owner_id=owner_id, taxonomy_name=taxonomy_name
            )
            if taxonomy:
                return self._extract_taxonomy_info(taxonomy)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Warning: Could not load taxonomy: {e}")
        return None

    def _build_generation_prompt(self, project_id: str, owner_id: str, 
                                prompt_name: str, chunk_text: str,
                                response_schema: Dict[str, Any],
                                extra_context: str = "") -> str:
        """Generic method for building generation prompts."""
        prompt_template = self._get_prompt_template(prompt_name, project_id, owner_id)
        param_dict = self.get_param_dict(project_id, owner_id)
        param_dict['chunk_text'] = chunk_text
        
        final_prompt = self._build_base_prompt(prompt_template, chunk_text, param_dict)
        
        if extra_context:
            final_prompt += f"\n\n{extra_context}"
        
        final_prompt += build_json_instructions(response_schema)
        
        return final_prompt

    def build_question_generation_prompt(self, project_id: str, owner_id: str, 
                                        chunk_text: str, taxonomy_project=None, 
                                        taxonomy_name=None) -> str:
        """Build prompt for question generation."""
        return self._build_generation_prompt(
            project_id, owner_id, "question_generation", chunk_text,
            QUESTION_GENERATION_SCHEMA
        )

    def build_answer_generation_prompt(self, project_id: str, owner_id: str, 
                                      chunk_text: str, taxonomy_project=None, 
                                      taxonomy_name=None) -> str:
        """Build prompt for answer generation."""
        return self._build_generation_prompt(
            project_id, owner_id, "answer_generation", chunk_text,
            ANSWER_GENERATION_SCHEMA
        )

    def build_summarization_prompt(self, project_id: str, owner_id: str, 
                                  chunk_text: str, taxonomy_project=None, 
                                  taxonomy_name=None) -> str:
        """Build prompt for summarization."""
        return self._build_generation_prompt(
            project_id, owner_id, "summarization", chunk_text,
            SUMMARIZATION_SCHEMA
        )

    def get_param_dict(self, project_id: str, owner_id: str) -> dict:
        """Get dataset parameters with defaults applied."""
        params_records = self.dataset_parameter_repository.get_parameters_by_project(project_id)
        param_dict = {}
        
        if params_records and isinstance(params_records, list):
            params = params_records[0]
            if isinstance(params, dict):
                param_dict = params.copy()
            elif isinstance(params, (tuple, list)):
                param_keys = ["id", "owner_id", "project_id", "purpose", "audience", 
                             "extraction_rules", "dataset_format", "question_style", 
                             "answer_style", "negativity_ratio", "data_augmentation", 
                             "custom_audience", "custom_purpose", "complexity_level", "domain"]
                param_dict = dict(zip(param_keys, params))
        
        # Apply defaults
        param_dict.setdefault("id", "default")
        param_dict.setdefault("owner_id", owner_id)
        param_dict.setdefault("project_id", project_id)
        
        for key, value in DEFAULT_DATASET_PARAMS.items():
            param_dict.setdefault(key, value)
        
        # Add computed context
        complexity_level = param_dict.get('complexity_level', 'intermediate')
        param_dict['complexity_level_context'] = (
            f' with {complexity_level} complexity level' 
            if complexity_level and complexity_level != 'auto' else ''
        )
        
        return param_dict

    def _extract_taxonomy_info(self, taxonomy) -> dict:
        """Extract taxonomy information into a dictionary."""
        info = {
            'taxonomy_name': taxonomy.name if hasattr(taxonomy, 'name') else 'unknown',
            'taxonomy_description': taxonomy.description if hasattr(taxonomy, 'description') else ''
        }
        
        categories = []
        def collect(cat, depth=0):
            if depth >= 3 or not cat:
                return
            if hasattr(cat, 'name'):
                categories.append(cat.name)
            if hasattr(cat, 'children'):
                for child in cat.children:
                    collect(child, depth + 1)
        
        collect(taxonomy)
        info['taxonomy_categories'] = ', '.join(categories)
        info['taxonomy_instructions'] = f"Use categories: {info['taxonomy_categories']}"
        return info

    def _build_simple_prompt(self, project_id: str, owner_id: str, 
                            prompt_name: str, chunk_text: str,
                            extra_params: Optional[dict] = None) -> str:
        """Build a simple prompt with optional extra parameters."""
        param_dict = self.get_param_dict(project_id, owner_id)
        param_dict['chunk_text'] = chunk_text
        
        if extra_params:
            param_dict.update(extra_params)
            
        prompt_record = self.prompt_repository.get_by_name(prompt_name, str(project_id), owner_id)
        
        if not prompt_record:
            raise ValueError(f"Prompt '{prompt_name}' not found.")
            
        template = prompt_record.get('content', '') if isinstance(prompt_record, dict) else str(prompt_record)
        return self._safe_format(template, param_dict)

    def build_instruction_following_prompt(self, project_id: str, owner_id: str, chunk_text: str) -> str:
        """Build prompt for instruction following generation."""
        return self._build_simple_prompt(project_id, owner_id, "instruction_following", chunk_text)

    def build_advanced_extraction_prompt(self, project_id: str, owner_id: str, chunk_text: str) -> str:
        """Build prompt for advanced extraction."""
        return self._build_simple_prompt(project_id, owner_id, "advanced_extraction", chunk_text)

    def build_classification_prompt(self, project_id: str, owner_id: str, 
                                   chunk_text: str, classification_schema: dict) -> str:
        """Build prompt for classification tasks."""
        return self._build_simple_prompt(
            project_id, owner_id, "classification", chunk_text,
            extra_params={'classification_schema': str(classification_schema)}
        )

    def build_custom_prompt(self, project_id: str, owner_id: str, 
                           custom_prompt: str, chunk_text: Optional[str] = None) -> str:
        """Build a prompt using a custom template string."""
        param_dict = self.get_param_dict(project_id, owner_id)
        final_prompt = custom_prompt
        
        if chunk_text is not None:
            if "{chunk}" in final_prompt:
                final_prompt = final_prompt.replace("{chunk}", chunk_text)
            else:
                final_prompt = f"Content: {chunk_text}\n\n{final_prompt}"
                
        return self._safe_format(final_prompt, param_dict)
