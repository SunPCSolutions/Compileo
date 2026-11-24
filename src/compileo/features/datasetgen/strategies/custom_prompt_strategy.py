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
        Call LLM once per chunk and parse all pairs from the single response.
        """
        results = []
        
        # Replace {chunk} placeholder with actual chunk text
        if "{chunk}" in custom_prompt:
            final_prompt = custom_prompt.replace("{chunk}", chunk)
        else:
            final_prompt = f"Content: {chunk}\n\n" + custom_prompt

        # Determine if AI-decides mode
        ai_decides = (datasets_per_chunk == 0)

        # Add clear JSON format instructions for the custom prompt
        json_instructions = """

IMPORTANT: You must respond with a valid JSON array of objects with exactly this structure:
[
  {
    "question": "The question you generated based on the content",
    "answer": "The answer to the question",
    "reasoning": "Brief explanation of your reasoning",
    "confidence_score": 0.8,
    "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
  }
]

CRITICAL CONSTRAINTS:
- ONLY use information explicitly present in the provided content.
- Generate questions that are directly answerable from the content provided.
- Base answers strictly on the content provided.
- Each question must cover a DIFFERENT topic, drug, or concept from the content.
- DO NOT generate multiple questions about the same drug, dose, or concept - each question must be truly unique.
- If the content contains multiple drugs, generate questions about different drugs, not the same drug repeatedly.
- If the content doesn't support generating meaningful questions and answers, return an empty array [].

Return ONLY the JSON array, no additional text or explanation."""

        if ai_decides:
            json_instructions += """\n\nGenerate as many question-answer pairs as appropriate based on the richness and diversity of the content.
Consider:
- The amount of distinct information in the text
- Different concepts, topics, or entities present
- Varying complexity levels suitable for different audiences
- Return 1-5 entries for sparse content, 5-10 for rich content, more for very comprehensive content.
Return them as a JSON array of objects."""
        elif datasets_per_chunk > 1:
            json_instructions += f"\n\nGenerate exactly {datasets_per_chunk} question-answer pairs covering DIFFERENT topics, drugs, or concepts from the content. Each question must be about a different aspect - do not repeat the same information in multiple questions. Return them as a JSON array of objects."

        final_prompt += json_instructions

        result = self.llm_interaction.generate(final_prompt)
        logger.debug(f"LLM result: {result} (type: {type(result)})")

        # Parse JSON string if needed
        if isinstance(result, str):
            try:
                result = json.loads(result.strip())
                logger.debug(f"Parsed JSON string to: {type(result)}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return results

        # Handle multiple results from single LLM call
        # For AI-determined mode (datasets_per_chunk == 0), don't limit results
        max_items = None if ai_decides else datasets_per_chunk
        
        if isinstance(result, list):
            items_to_process = result if max_items is None else result[:max_items]
            for item in items_to_process:
                if item and isinstance(item, dict):
                    if ai_decides:
                        item['ai_determined'] = True
                    results.append(item)
                    logger.debug(f"Added list item: {item.get('question', 'no question')[:50]}...")
                else:
                    logger.warning(f"Skipping invalid list item: {item}")
        elif isinstance(result, dict) and result:
            if ai_decides:
                result['ai_determined'] = True
            results.append(result)
            logger.debug(f"Added dict result: {result.get('question', 'no question')[:50]}...")
        else:
            logger.warning(f"Skipping invalid result: {result}")

        # Only apply limit for fixed-count mode
        if ai_decides:
            logger.info(f"AI-determined mode generated {len(results)} entries from custom prompt")
            return results
        return results[:datasets_per_chunk]
