from typing import List, Dict, Any, Optional
from src.compileo.core.logging import get_logger
from .base_strategy import BaseGenerationStrategy
from ..constants import build_json_instructions

logger = get_logger(__name__)


class ChunkGenerationStrategy(BaseGenerationStrategy):
    """Strategy for generating datasets from raw text chunks."""

    def generate_qa(self, project_id: str, owner_id: str, prompt_name: str, chunk: str,
                    taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                    datasets_per_chunk: int = 3, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate Q&A pairs for a single chunk."""
        final_prompt = self.prompt_builder.build_prompt(
            project_id, owner_id, prompt_name, chunk,
            taxonomy_project=taxonomy_project,
            taxonomy_name=taxonomy_name,
            datasets_per_chunk=datasets_per_chunk
        )

        # Modify prompt to request multiple items if needed (only for fixed counts)
        if datasets_per_chunk > 1:
            final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs from this text chunk. Return them as a JSON array."

        result = self.llm_interaction.generate(final_prompt, options=options)
        source_type = "taxonomy_filtered_chunk" if taxonomy_project and taxonomy_name else "chunk"
        return self._process_result(result, source_type, datasets_per_chunk)

    def generate_questions(self, project_id: str, chunk: str, owner_id: str,
                          taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                          datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """Generate only questions for a single chunk."""
        final_prompt = self.prompt_builder.build_question_generation_prompt(
            project_id, owner_id, chunk, taxonomy_project, taxonomy_name
        )

        # For AI-determined mode, add instructions to generate multiple questions
        if datasets_per_chunk == 0:
            from ..constants import QUESTION_GENERATION_SCHEMA
            final_prompt += build_json_instructions(
                {"type": "array", "items": QUESTION_GENERATION_SCHEMA, "description": "Array of question-generation results. Generate as many as appropriate."},
                ai_decides=True
            )
        elif datasets_per_chunk > 1:
            final_prompt += f"\n\nGenerate {datasets_per_chunk} different sets of questions from this text chunk. Return them as a JSON array."

        result = self.llm_interaction.generate(final_prompt)
        return self._process_result(result, "chunk_questions", datasets_per_chunk)

    def generate_answers(self, project_id: str, chunk: str, owner_id: str,
                        taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                        datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """Generate only answers for a single chunk."""
        final_prompt = self.prompt_builder.build_answer_generation_prompt(
            project_id, owner_id, chunk, taxonomy_project, taxonomy_name
        )

        # For AI-determined mode, add instructions to generate multiple answers
        if datasets_per_chunk == 0:
            from ..constants import ANSWER_GENERATION_SCHEMA
            final_prompt += build_json_instructions(
                {"type": "array", "items": ANSWER_GENERATION_SCHEMA, "description": "Array of answer-generation results. Generate as many as appropriate."},
                ai_decides=True
            )
        elif datasets_per_chunk > 1:
            final_prompt += f"\n\nGenerate {datasets_per_chunk} different sets of answers from this text chunk. Return them as a JSON array."

        result = self.llm_interaction.generate(final_prompt)
        return self._process_result(result, "chunk_answers", datasets_per_chunk)

    def generate_summary(self, project_id: str, chunk: str, owner_id: str,
                        taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                        datasets_per_chunk: int = 1) -> List[Dict[str, Any]]:
        """Generate content summaries for a single chunk."""
        final_prompt = self.prompt_builder.build_summarization_prompt(
            project_id, owner_id, chunk, taxonomy_project, taxonomy_name
        )

        # For AI-determined mode, add instructions to generate multiple summaries
        if datasets_per_chunk == 0:
            from ..constants import SUMMARIZATION_SCHEMA
            final_prompt += build_json_instructions(
                {"type": "array", "items": SUMMARIZATION_SCHEMA, "description": "Array of summaries. Generate as many as appropriate."},
                ai_decides=True
            )
        elif datasets_per_chunk > 1:
            final_prompt += f"\n\nGenerate {datasets_per_chunk} different summaries from this text chunk. Return them as a JSON array."

        result = self.llm_interaction.generate(final_prompt)
        return self._process_result(result, "chunk_summary", datasets_per_chunk)
