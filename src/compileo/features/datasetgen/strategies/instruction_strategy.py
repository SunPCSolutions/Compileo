from typing import List, Dict, Any, Optional
import random
from src.compileo.core.logging import get_logger
from .base_strategy import BaseGenerationStrategy
from ..constants import INSTRUCTION_RESPONSE_SCHEMA, build_json_instructions

logger = get_logger(__name__)


class InstructionGenerationStrategy(BaseGenerationStrategy):
    """Strategy for generating instruction-following datasets."""

    INSTRUCTION_TYPES = [
        "explain", "describe", "analyze", "summarize",
        "create", "implement", "design", "troubleshoot"
    ]

    def generate_instruction_following(self, project_id: str, chunk: str,
                                     taxonomy_project: Optional[str] = None,
                                     taxonomy_name: Optional[str] = None,
                                     datasets_per_chunk: int = 3,
                                     owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generates instruction-response pairs for a single chunk."""
        results = []

        # For AI-determined mode (datasets_per_chunk == 0), make a single call with array schema
        if datasets_per_chunk == 0:
            from ..constants import INSTRUCTION_RESPONSE_ARRAY_SCHEMA
            
            instruction_type = random.choice(self.INSTRUCTION_TYPES)
            final_prompt = self.build_instruction_following_prompt(
                chunk, instruction_type, taxonomy_project, taxonomy_name,
                ai_decides=True
            )
            
            result = self.llm_interaction.generate(final_prompt)
            
            # Process all results from AI-determined response (max_items=0 means no limit)
            processed = self._process_result(result, "instruction_following", 0,
                                           instruction_type=instruction_type)
            
            # Filter to ensure valid instruction-response format
            valid_items = [
                item for item in processed
                if isinstance(item, dict) and 'instruction' in item and 'output' in item
            ]
            results.extend(valid_items)
            logger.info(f"AI-determined mode generated {len(results)} instruction-following entries")
            return results

        # Fixed count mode: loop for specified number of items
        for i in range(datasets_per_chunk):
            instruction_type = random.choice(self.INSTRUCTION_TYPES)
            final_prompt = self.build_instruction_following_prompt(
                chunk, instruction_type, taxonomy_project, taxonomy_name
            )

            result = self.llm_interaction.generate(final_prompt)
            processed = self._process_result(result, "instruction_following", 1,
                                           instruction_type=instruction_type)
            
            # Filter to ensure valid instruction-response format
            valid_items = [
                item for item in processed
                if isinstance(item, dict) and 'instruction' in item and 'output' in item
            ]
            results.extend(valid_items)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def build_instruction_following_prompt(self, chunk: str, instruction_type: str,
                                         taxonomy_project: Optional[str] = None,
                                         taxonomy_name: Optional[str] = None,
                                         ai_decides: bool = False) -> str:
        """Builds a prompt for generating instruction-response pairs."""
        
        if ai_decides:
            base_prompt = f"""Generate instruction-response pairs based on the following content.

Content: {chunk}

Instruction type: {instruction_type}"""
        else:
            base_prompt = f"""Generate an instruction-response pair based on the following content.

Content: {chunk}

Instruction type: {instruction_type}"""

        # Add taxonomy context if available
        if taxonomy_name:
            base_prompt += f"\n\nContext: This content relates to the taxonomy category '{taxonomy_name}'."

        # Add standardized instructions
        if ai_decides:
            from ..constants import INSTRUCTION_RESPONSE_ARRAY_SCHEMA
            base_prompt += build_json_instructions(INSTRUCTION_RESPONSE_ARRAY_SCHEMA, ai_decides=True)
            base_prompt += """

Generate as many instruction-response pairs as appropriate based on the content richness.
The instructions should be things someone would actually ask about this content.
The responses should be detailed and directly answer each instruction."""
        else:
            base_prompt += build_json_instructions(INSTRUCTION_RESPONSE_SCHEMA)
            base_prompt += """

The instruction should be something someone would actually ask about this content.
The response should be detailed and directly answer the instruction."""

        return base_prompt
