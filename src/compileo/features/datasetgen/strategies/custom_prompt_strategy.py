from typing import List, Dict, Any, Optional
import json
from src.compileo.core.logging import get_logger
from .base_strategy import BaseGenerationStrategy

logger = get_logger(__name__)

class CustomPromptStrategy(BaseGenerationStrategy):
    """
    Strategy for generating datasets using custom user-provided prompts.
    """

    def generate_with_custom_prompt(self, custom_prompt: str, chunk: str,
                                    taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                    datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates content using a custom prompt.
        """
        results = []
        for i in range(datasets_per_chunk):
            # Replace {chunk} placeholder with actual chunk text
            if "{chunk}" in custom_prompt:
                final_prompt = custom_prompt.replace("{chunk}", chunk)
            else:
                final_prompt = f"Content: {chunk}\n\n" + custom_prompt

            # Add clear JSON format instructions for the custom prompt
            json_instructions = f"""

IMPORTANT: You must respond with a valid JSON object (or array of objects) with exactly this structure:
{{
  "question": "The question you generated based on the content",
  "answer": "The answer to the question",
  "reasoning": "Brief explanation of your reasoning",
  "confidence_score": 0.8,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}}

CRITICAL CONSTRAINTS:
- ONLY use information explicitly present in the provided content.
- Generate questions that are directly answerable from the content provided.
- Base answers strictly on the content provided.
- If the content doesn't support generating meaningful questions and answers, return an empty array [].

Return ONLY the JSON object or array, no additional text or explanation."""

            if datasets_per_chunk > 1:
                json_instructions += f"\n\nGenerate {datasets_per_chunk} different clinical case question-answer pairs. Return them as a JSON array of objects."

            final_prompt += json_instructions

            result = self.llm_interaction.generate(final_prompt)
            logger.debug(f"LLM result for chunk {i+1}: {result} (type: {type(result)})")

            # Parse JSON string if needed
            if isinstance(result, str):
                try:
                    result = json.loads(result.strip())
                    logger.debug(f"Parsed JSON string to: {type(result)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    continue

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    if item and isinstance(item, dict):
                        results.append(item)
                        logger.debug(f"Added list item: {item.get('question', 'no question')[:50]}...")
                    else:
                        logger.warning(f"Skipping invalid list item: {item}")
            elif isinstance(result, dict) and result:
                results.append(result)
                logger.debug(f"Added dict result: {result.get('question', 'no question')[:50]}...")
            else:
                logger.warning(f"Skipping invalid result: {result}")

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]
