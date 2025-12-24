from typing import Optional
from src.compileo.storage.src.project.database_repositories import PromptRepository, DatasetParameterRepository
from ...core.logging import get_logger

logger = get_logger(__name__)

class PromptBuilder:
    """
    Builds prompts by combining templates and dataset parameters.
    """
    def __init__(self, prompt_repository: PromptRepository, dataset_parameter_repository: DatasetParameterRepository):
        """
        Initializes the PromptBuilder.

        Args:
            prompt_repository: The repository for accessing prompt templates.
            dataset_parameter_repository: The repository for accessing dataset parameters.
        """
        self.prompt_repository = prompt_repository
        self.dataset_parameter_repository = dataset_parameter_repository

    def build_prompt(self, project_id: int, prompt_name: str, chunk_text: Optional[str] = None,
                    taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                    datasets_per_chunk: int = 3) -> str:
        """
        Builds a prompt by combining a template with dataset parameters.

        Args:
            project_id: The ID of the project.
            prompt_name: The name of the prompt template to use.
            chunk_text: The text chunk to insert (optional).

        Returns:
            The generated prompt with parameters injected.

        Raises:
            ValueError: If the prompt or dataset parameters are not found.
        """
        prompt_record = self.prompt_repository.get_by_name(prompt_name)
        if prompt_record:
            prompt_template = prompt_record[2]  # content is the third column (index 2)
        elif prompt_name == "default":
            # Hardcoded default prompt template for dataset generation
            prompt_template = """Generate {datasets_per_chunk} question-answer pairs from the following content.

Content: {chunk}

Instructions:
- Create diverse, high-quality questions that test understanding of the content
- Provide accurate, comprehensive answers based only on the given content
- Ensure questions are answerable using only the provided information
- Vary question types (factual, analytical, applicative)
- Focus on key concepts, important details, and relationships in the content

{purpose_context}{audience_context}{complexity_level_context}

Format your response as a JSON array of objects with this structure:
[
  {{
    "question": "Your question here",
    "answer": "Your answer here",
    "category": "content_category",
    "difficulty": "easy|medium|hard"
  }}
]"""
        else:
            raise ValueError(f"Prompt '{prompt_name}' not found.")

        # Get parameter dictionary, with fallback for default prompt
        try:
            param_dict = self.get_param_dict(project_id)
        except ValueError:
            if prompt_name == "default":
                # Fallback parameters for default prompt when dataset parameters don't exist
                param_dict = {
                    'datasets_per_chunk': datasets_per_chunk,
                    'purpose_context': '',
                    'audience_context': '',
                    'complexity_level_context': ' with intermediate complexity level'
                }
            else:
                raise

        # Ensure datasets_per_chunk is in param_dict
        param_dict['datasets_per_chunk'] = datasets_per_chunk

        # Load taxonomy if specified
        taxonomy_info = None
        if taxonomy_project and taxonomy_name:
            from src.compileo.features.taxonomy.loader import TaxonomyLoader
            from src.compileo.storage.src.project.database_repositories import ProcessedOutputRepository
            # Note: In a real implementation, we'd inject the repo, but for now we'll create it
            # This is a temporary solution - in production, this should be injected
            import sqlite3
            from src.compileo.storage.src.database import get_db_connection
            db_conn = get_db_connection()
            processed_repo = ProcessedOutputRepository(db_conn)
            taxonomy_loader = TaxonomyLoader(processed_repo)
            try:
                taxonomy = taxonomy_loader.load_taxonomy_by_project(taxonomy_project, taxonomy_name)
                if taxonomy:
                    taxonomy_info = self._extract_taxonomy_info(taxonomy, param_dict)
            except Exception as e:
                # Log warning but continue without taxonomy
                logger.warning(f"Could not load taxonomy {taxonomy_project}/{taxonomy_name}: {e}")

        # Add taxonomy info to param_dict if available
        if taxonomy_info:
            param_dict.update(taxonomy_info)

        # Replace {chunk} with actual chunk text if provided
        if chunk_text is not None:
            if "{chunk}" in prompt_template:
                final_prompt = prompt_template.replace("{chunk}", chunk_text)
            else:
                # Fallback: Prepend content if placeholder is missing
                final_prompt = f"Content: {chunk_text}\n\n{prompt_template}"
        else:
            final_prompt = prompt_template

        # Format with parameters (avoiding {chunk} conflicts)
        final_prompt = final_prompt.format(**param_dict)

        # Append reasoning metadata instructions for enhanced dataset generation
        reasoning_instructions = """

CRITICAL CONSTRAINTS:
- ONLY use information and concepts that are explicitly present in the provided text chunk above.
- DO NOT draw from external knowledge, general knowledge, or information not contained in the text.
- DO NOT generate synthetic or hypothetical content.
- If the text does not contain sufficient information for a response, do not generate one.
- Base all content strictly on the information provided in the text chunk.

IMPORTANT: Provide your response as a valid machine-readable structure with the following fields:
{
  "question": "The question you generated based on the content",
  "answer": "The answer to the question",
  "reasoning": "Brief explanation of your reasoning process",
  "confidence_score": 0.0-1.0,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}

Return ONLY the raw object, no additional text."""

        final_prompt += reasoning_instructions

        return final_prompt

    def get_param_dict(self, project_id: int) -> dict:
        """
        Retrieves and formats dataset parameters into a dictionary.

        Args:
            project_id: The ID of the project.

        Returns:
            A dictionary of dataset parameters.
        
        Raises:
            ValueError: If dataset parameters for the project are not found.
        """
        params_records = self.dataset_parameter_repository.get_by_project_id(project_id)
        if not params_records:
            raise ValueError(f"Dataset parameters for project ID '{project_id}' not found.")

        # Using the first set of parameters found for the project.
        if isinstance(params_records, list) and params_records:
            params = params_records[0]  # Get the first row
        else:
            params = params_records  # Single row

        # Column names from the 'dataset_parameters' table schema.
        param_keys = [
            "id", "project_id", "purpose", "audience", "extraction_rules",
            "dataset_format", "question_style", "answer_style",
            "negativity_ratio", "data_augmentation", "custom_audience",
            "custom_purpose", "complexity_level", "domain"
        ]

        # Convert Row object to dict or use existing dict
        if isinstance(params, dict):
            param_dict = params
        elif hasattr(params, '__getitem__'):
            # Handle Row or tuple by index
            try:
                params_tuple = tuple(params[i] for i in range(len(param_keys)))
                param_dict = dict(zip(param_keys, params_tuple))
            except (KeyError, IndexError):
                # Handle Row by key if index fails
                param_dict = {key: params[key] for key in param_keys if key in params}
        else:
            param_dict = {}

        # Provide defaults for new fields if they don't exist (backward compatibility)
        param_dict.setdefault('custom_audience', '')
        param_dict.setdefault('custom_purpose', '')
        param_dict.setdefault('complexity_level', 'intermediate')
        param_dict.setdefault('domain', 'general')

        # Handle optional complexity level - create context string for prompts
        complexity_level = param_dict.get('complexity_level', 'intermediate')
        if complexity_level == 'auto' or not complexity_level:
            param_dict['complexity_level_context'] = ''
        else:
            param_dict['complexity_level_context'] = f' with {complexity_level} complexity level'

        # Ensure context parameters exist (for template compatibility)
        param_dict.setdefault('purpose_context', '')
        param_dict.setdefault('audience_context', '')

        return param_dict

    def build_question_generation_prompt(self, project_id: int, chunk_text: str,
                                       taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                       datasets_per_chunk: int = 3) -> str:
        """
        Builds a prompt for generating questions from a text chunk.

        Args:
            project_id: The ID of the project.
            chunk_text: The text chunk to generate questions from.
            datasets_per_chunk: Number of questions to generate.

        Returns:
            A string containing the formatted prompt for question generation.
        """
        param_dict = self.get_param_dict(project_id)
        param_dict['datasets_per_chunk'] = datasets_per_chunk

        # Load taxonomy if specified
        if taxonomy_project and taxonomy_name:
            from src.compileo.features.taxonomy.loader import TaxonomyLoader
            from src.compileo.storage.src.project.database_repositories import ProcessedOutputRepository
            import sqlite3
            from src.compileo.storage.src.database import get_db_connection
            db_conn = get_db_connection()
            processed_repo = ProcessedOutputRepository(db_conn)
            taxonomy_loader = TaxonomyLoader(processed_repo)
            try:
                taxonomy = taxonomy_loader.load_taxonomy_by_project(taxonomy_project, taxonomy_name)
                if taxonomy:
                    taxonomy_info = self._extract_taxonomy_info(taxonomy, param_dict)
                    param_dict.update(taxonomy_info)
            except Exception as e:
                logger.warning(f"Could not load taxonomy {taxonomy_project}/{taxonomy_name}: {e}")

        prompt_record = self.prompt_repository.get_by_name("question_generation")
        if prompt_record:
            prompt_template = prompt_record[2]
        else:
            # Fallback template
            prompt_template = "Generate a question based on the following content: {chunk_text}"

        # Format the prompt with the parameters and the chunk text
        final_prompt = prompt_template.format(
            chunk_text=chunk_text,
            **param_dict
        )

        # Ensure chunk text is included
        if chunk_text not in final_prompt:
            final_prompt = f"Content: {chunk_text}\n\n{final_prompt}"

        # Append reasoning metadata instructions
        reasoning_instructions = """

CRITICAL CONSTRAINTS:
- ONLY use information and concepts that are explicitly present in the provided text chunk above.
- DO NOT draw from external knowledge, general knowledge, or information not contained in the text.
- DO NOT generate synthetic or hypothetical content.
- If the text does not contain sufficient information for questions, do not generate them.
- Base all questions strictly on the content provided in the text chunk.

IMPORTANT: Provide your response as a valid machine-readable structure with the following fields:
{
  "question": "The question you generated based on the content",
  "reasoning": "Brief explanation of your reasoning process",
  "confidence_score": 0.0-1.0,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}

Return ONLY the raw object, no additional text."""

        final_prompt += reasoning_instructions

        return final_prompt

    def build_answer_generation_prompt(self, project_id: int, chunk_text: str,
                                     taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                     datasets_per_chunk: int = 3) -> str:
        """
        Builds a prompt for generating answers from a text chunk.

        Args:
            project_id: The ID of the project.
            chunk_text: The text chunk to generate answers from.
            datasets_per_chunk: Number of answers to generate.

        Returns:
            A string containing the formatted prompt for answer generation.
        """
        param_dict = self.get_param_dict(project_id)
        param_dict['datasets_per_chunk'] = datasets_per_chunk

        # Load taxonomy if specified
        if taxonomy_project and taxonomy_name:
            from src.compileo.features.taxonomy.loader import TaxonomyLoader
            from src.compileo.storage.src.project.database_repositories import ProcessedOutputRepository
            import sqlite3
            from src.compileo.storage.src.database import get_db_connection
            db_conn = get_db_connection()
            processed_repo = ProcessedOutputRepository(db_conn)
            taxonomy_loader = TaxonomyLoader(processed_repo)
            try:
                taxonomy = taxonomy_loader.load_taxonomy_by_project(taxonomy_project, taxonomy_name)
                if taxonomy:
                    taxonomy_info = self._extract_taxonomy_info(taxonomy, param_dict)
                    param_dict.update(taxonomy_info)
            except Exception as e:
                logger.warning(f"Could not load taxonomy {taxonomy_project}/{taxonomy_name}: {e}")

        prompt_record = self.prompt_repository.get_by_name("answer_generation")
        if prompt_record:
            prompt_template = prompt_record[2]
        else:
            # Fallback template
            prompt_template = "Generate an answer for the following content: {chunk_text}"

        # Format the prompt with the parameters and the chunk text
        final_prompt = prompt_template.format(
            chunk_text=chunk_text,
            **param_dict
        )

        # Ensure chunk text is included
        if chunk_text not in final_prompt:
            final_prompt = f"Content: {chunk_text}\n\n{final_prompt}"

        # Append reasoning metadata instructions
        reasoning_instructions = """

CRITICAL CONSTRAINTS:
- ONLY use information and concepts that are explicitly present in the provided text chunk above.
- DO NOT draw from external knowledge, general knowledge, or information not contained in the text.
- DO NOT generate synthetic or hypothetical content.
- If the text does not contain sufficient information for answers, do not generate them.
- Base all answers strictly on the content provided in the text chunk.

IMPORTANT: Provide your response as a valid machine-readable structure with the following fields:
{
  "answer": "The answer you generated based on the content",
  "reasoning": "Brief explanation of your reasoning process",
  "confidence_score": 0.0-1.0,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}

Return ONLY the raw object, no additional text."""

        final_prompt += reasoning_instructions

        return final_prompt

    def build_summarization_prompt(self, project_id: int, chunk_text: str,
                                 taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                 datasets_per_chunk: int = 3) -> str:
        """
        Builds a prompt for generating summaries from a text chunk.

        Args:
            project_id: The ID of the project.
            chunk_text: The text chunk to summarize.
            datasets_per_chunk: Number of summaries to generate.

        Returns:
            A string containing the formatted prompt for summarization.
        """
        param_dict = self.get_param_dict(project_id)
        param_dict['datasets_per_chunk'] = datasets_per_chunk

        # Load taxonomy if specified
        if taxonomy_project and taxonomy_name:
            from src.compileo.features.taxonomy.loader import TaxonomyLoader
            from src.compileo.storage.src.project.database_repositories import ProcessedOutputRepository
            import sqlite3
            from src.compileo.storage.src.database import get_db_connection
            db_conn = get_db_connection()
            processed_repo = ProcessedOutputRepository(db_conn)
            taxonomy_loader = TaxonomyLoader(processed_repo)
            try:
                taxonomy = taxonomy_loader.load_taxonomy_by_project(taxonomy_project, taxonomy_name)
                if taxonomy:
                    taxonomy_info = self._extract_taxonomy_info(taxonomy, param_dict)
                    param_dict.update(taxonomy_info)
            except Exception as e:
                logger.warning(f"Could not load taxonomy {taxonomy_project}/{taxonomy_name}: {e}")

        prompt_record = self.prompt_repository.get_by_name("summarization")
        if prompt_record:
            prompt_template = prompt_record[2]
        else:
            # Fallback template
            prompt_template = "Summarize the following content in a concise manner: {chunk_text}"

        # Format the prompt with the parameters and the chunk text
        final_prompt = prompt_template.format(
            chunk_text=chunk_text,
            **param_dict
        )

        # Ensure chunk text is included
        if chunk_text not in final_prompt:
            final_prompt = f"Content: {chunk_text}\n\n{final_prompt}"

        # Append reasoning metadata instructions
        reasoning_instructions = """

CRITICAL CONSTRAINTS:
- ONLY use information and concepts that are explicitly present in the provided text chunk above.
- DO NOT draw from external knowledge, general knowledge, or information not contained in the text.
- DO NOT generate synthetic or hypothetical content.
- If the text does not contain sufficient information for summarization, do not generate one.
- Base all summaries strictly on the content provided in the text chunk.

IMPORTANT: Provide your response as a valid machine-readable structure with the following fields:
{
  "summary": "Concise summary of the text",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "reasoning": "Brief explanation of your reasoning process",
  "confidence_score": 0.0-1.0,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}

Return ONLY the raw object, no additional text."""

        final_prompt += reasoning_instructions

        return final_prompt

    def _extract_taxonomy_info(self, taxonomy, param_dict: dict) -> dict:
        """
        Extract taxonomy information for prompt incorporation based on purpose and audience.

        Args:
            taxonomy: HierarchicalCategory taxonomy object
            param_dict: Current parameter dictionary with purpose and audience

        Returns:
            Dictionary with taxonomy-related parameters for prompt formatting
        """
        purpose = param_dict.get('purpose', '').lower()
        audience = param_dict.get('audience', '').lower()

        taxonomy_info = {
            'taxonomy_name': taxonomy.name,
            'taxonomy_description': taxonomy.description or '',
            'taxonomy_categories': self._flatten_taxonomy_categories(taxonomy)
        }

        # Adapt content based on purpose and audience
        if 'clinical' in purpose or 'diagnosis' in purpose:
            if 'expert' in audience:
                taxonomy_info['taxonomy_focus'] = 'clinical_diagnosis_expert'
                taxonomy_info['taxonomy_instructions'] = (
                    "Focus on differential diagnosis, pathophysiology, and evidence-based clinical reasoning. "
                    "Use medical terminology and consider comorbidities."
                )
            elif 'novice' in audience or 'student' in audience:
                taxonomy_info['taxonomy_focus'] = 'clinical_diagnosis_novice'
                taxonomy_info['taxonomy_instructions'] = (
                    "Explain diagnoses using simple language, focus on key symptoms and basic pathophysiology. "
                    "Avoid complex medical jargon or provide explanations."
                )
            else:
                taxonomy_info['taxonomy_focus'] = 'clinical_diagnosis_general'
                taxonomy_info['taxonomy_instructions'] = (
                    "Provide balanced clinical information suitable for healthcare professionals."
                )
        else:
            # Default taxonomy integration
            taxonomy_info['taxonomy_focus'] = 'general'
            taxonomy_info['taxonomy_instructions'] = (
                f"Use the taxonomy categories: {taxonomy_info['taxonomy_categories']} "
                "to structure your response appropriately."
            )

        return taxonomy_info

    def validate_taxonomy(self, taxonomy_project: str, taxonomy_name: str) -> bool:
        """
        Validate that the specified taxonomy exists and is accessible.

        Args:
            taxonomy_project: Project name containing the taxonomy
            taxonomy_name: Name of the taxonomy

        Returns:
            True if taxonomy exists and is valid, False otherwise
        """
        if not taxonomy_project or not taxonomy_name:
            return False

        try:
            from src.compileo.features.taxonomy.loader import TaxonomyLoader
            from src.compileo.storage.src.project.database_repositories import ProcessedOutputRepository
            import sqlite3
            from src.compileo.storage.src.database import get_db_connection
            db_conn = get_db_connection()
            processed_repo = ProcessedOutputRepository(db_conn)
            taxonomy_loader = TaxonomyLoader(processed_repo)
            taxonomy = taxonomy_loader.load_taxonomy_by_project(taxonomy_project, taxonomy_name)
            return taxonomy is not None
        except Exception as e:
            logger.error(f"Taxonomy validation failed: {e}")
            return False

    def _flatten_taxonomy_categories(self, taxonomy, max_depth: int = 3) -> str:
        """
        Flatten taxonomy categories into a readable string.

        Args:
            taxonomy: HierarchicalCategory taxonomy object
            max_depth: Maximum depth to traverse

        Returns:
            Comma-separated string of category names
        """
        categories = []

        def collect_categories(cat, depth=0):
            if depth >= max_depth:
                return
            categories.append(cat.name)
            for child in cat.children:
                collect_categories(child, depth + 1)

        collect_categories(taxonomy)
        return ', '.join(categories)

    def build_advanced_extraction_prompt(self, project_id: int, chunk_text: str) -> str:
        """
        Builds a prompt for advanced data extraction from a text chunk.

        Args:
            project_id: The ID of the project.
            chunk_text: The text chunk to be processed.

        Returns:
            A string containing the formatted prompt for the LLM.
        
        Raises:
            ValueError: If the prompt or dataset parameters are not found.
        """
        param_dict = self.get_param_dict(project_id)
        
        prompt_template = self.prompt_repository.get_by_name("advanced_extraction")
        if not prompt_template:
            raise ValueError("Prompt 'advanced_extraction' not found.")

        # Format the prompt with the parameters and the chunk text
        final_prompt = prompt_template.format(
            chunk_text=chunk_text,
            **param_dict
        )

        return final_prompt

    def build_classification_prompt(self, project_id: int, chunk_text: str, classification_schema: dict) -> str:
        """
        Builds a prompt for classifying a text chunk based on a schema.

        Args:
            project_id: The ID of the project.
            chunk_text: The text chunk to be classified.
            classification_schema: A dictionary defining the classification schema.

        Returns:
            A string containing the formatted prompt for the LLM.
        
        Raises:
            ValueError: If the prompt or dataset parameters are not found.
        """
        param_dict = self.get_param_dict(project_id)
        
        prompt_template = self.prompt_repository.get_by_name("classification")
        if not prompt_template:
            raise ValueError("Prompt 'classification' not found.")

        # Format the prompt with the parameters, chunk text, and schema
        final_prompt = prompt_template.format(
            chunk_text=chunk_text,
            classification_schema=classification_schema,
            **param_dict
        )

        return final_prompt