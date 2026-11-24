from typing import List, Dict, Any, Optional
import json
from src.compileo.core.logging import get_logger
from .base_strategy import BaseGenerationStrategy
from ..constants import (
    QA_RESPONSE_SCHEMA, QA_RESPONSE_ARRAY_SCHEMA,
    INSTRUCTION_RESPONSE_SCHEMA, INSTRUCTION_RESPONSE_ARRAY_SCHEMA,
    build_json_instructions
)

logger = get_logger(__name__)


class CustomPromptStrategy(BaseGenerationStrategy):
    """Strategy for generating datasets using custom prompts."""

    def generate_with_custom_prompt(self, project_id: str, chunk: str, owner_id: str,
                                   custom_prompt: str,
                                   generation_mode: str = "question and answer",
                                   datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """Generates content using a custom prompt for a single chunk."""
        # Replace {chunk} placeholder with actual chunk text
        if "{chunk}" in custom_prompt:
            final_prompt = custom_prompt.replace("{chunk}", chunk)
        else:
            final_prompt = f"Content: {chunk}\n\n" + custom_prompt

        # Determine the appropriate schema based on generation mode
        if "instruction" in generation_mode.lower():
            single_schema = INSTRUCTION_RESPONSE_SCHEMA
            array_schema = INSTRUCTION_RESPONSE_ARRAY_SCHEMA
        else:
            single_schema = QA_RESPONSE_SCHEMA
            array_schema = QA_RESPONSE_ARRAY_SCHEMA

        # Add standardized JSON instructions with the correct array vs object schema
        if datasets_per_chunk == 0:
            final_prompt += build_json_instructions(array_schema, ai_decides=True)
        elif datasets_per_chunk == 1:
            final_prompt += build_json_instructions(single_schema, multiple_items=1)
        else:
            final_prompt += build_json_instructions(array_schema, multiple_items=datasets_per_chunk)
        
        # Add diversity constraint for multiple items
        if datasets_per_chunk > 1:
            final_prompt += "\n\nEach question must cover a DIFFERENT topic, drug, or concept from the content. DO NOT generate multiple questions about the same drug, dose, or concept."

        result = self.llm_interaction.generate(final_prompt)
        
        # Parse JSON string if needed
        if isinstance(result, str):
            try:
                result = json.loads(result.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from custom prompt result: {e}")
                return []

        return self._process_result(result, "custom_prompt", datasets_per_chunk, 
                                   generation_mode=generation_mode)
