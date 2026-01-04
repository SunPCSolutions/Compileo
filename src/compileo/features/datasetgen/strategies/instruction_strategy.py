from typing import List, Dict, Any, Optional
import random
from src.compileo.core.logging import get_logger
from .base_strategy import BaseGenerationStrategy

logger = get_logger(__name__)

class InstructionGenerationStrategy(BaseGenerationStrategy):
    """
    Strategy for generating instruction-following datasets.
    """

    def generate_instruction_following(self, project_id: int, chunk: str,
                                     taxonomy_project: Optional[str] = None,
                                     taxonomy_name: Optional[str] = None,
                                     datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates instruction-response pairs for a single chunk.
        """
        results = []

        # Define instruction types based on research
        instruction_types = [
            "explain", "describe", "analyze", "summarize",
            "create", "implement", "design", "troubleshoot"
        ]

        for i in range(datasets_per_chunk):
            # Select random instruction type for diversity
            instruction_type = random.choice(instruction_types)

            # Build prompt for instruction generation
            final_prompt = self.build_instruction_following_prompt(
                chunk, instruction_type, taxonomy_project, taxonomy_name
            )

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:1]:  # Take only first result per generation
                    if isinstance(item, dict) and 'instruction' in item and 'output' in item:
                        item["instruction_type"] = instruction_type
                        item["source_type"] = "instruction_following"
                        results.append(item)
            else:
                if isinstance(result, dict) and 'instruction' in result and 'output' in result:
                    result["instruction_type"] = instruction_type
                    result["source_type"] = "instruction_following"
                    results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def build_instruction_following_prompt(self, chunk: str, instruction_type: str,
                                         taxonomy_project: Optional[str] = None,
                                         taxonomy_name: Optional[str] = None) -> str:
        """
        Builds a prompt for generating instruction-response pairs.
        """
        # Base prompt structure following Alpaca format
        base_prompt = f"""Generate an instruction-response pair based on the following content.

        Content: {chunk}

        Instruction type: {instruction_type}

CRITICAL CONSTRAINTS:
- ONLY use information and concepts that are explicitly present in the provided content above.
- DO NOT draw from external knowledge, general knowledge, or information not contained in the content.
- DO NOT generate synthetic or hypothetical content.
- If the content does not contain sufficient information for an instruction-response pair, do not generate one.
- Base all instructions and responses strictly on the content provided.

Return your response as a valid JSON object with exactly this structure:
{{
  "instruction": "A clear, actionable instruction related to the content",
  "input": "",
  "output": "A comprehensive, helpful response to the instruction"
}}

The instruction should be something someone would actually ask about this content.
The response should be detailed and directly answer the instruction.
"""

        # Add taxonomy context if available
        if taxonomy_name:
            base_prompt += f"\n\nContext: This content relates to the taxonomy category '{taxonomy_name}'."

        return base_prompt
