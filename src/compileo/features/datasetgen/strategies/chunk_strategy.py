from typing import List, Dict, Any, Optional
from src.compileo.core.logging import get_logger
from .base_strategy import BaseGenerationStrategy

logger = get_logger(__name__)

class ChunkGenerationStrategy(BaseGenerationStrategy):
    """
    Strategy for generating datasets from raw text chunks.
    """

    def generate_qa(self, project_id: int, prompt_name: str, chunk: str,
                    taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                    datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates multiple question-answer pairs for a single chunk.
        """
        results = []
        for i in range(datasets_per_chunk):
            final_prompt = self.prompt_builder.build_prompt(project_id, prompt_name, chunk,
                                                            taxonomy_project=taxonomy_project, taxonomy_name=taxonomy_name,
                                                            datasets_per_chunk=datasets_per_chunk)
            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs from this text chunk. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    results.append(item)
            else:
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def generate_questions(self, project_id: int, chunk: str,
                           taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                           datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates multiple questions for a single chunk.
        """
        results = []
        for i in range(datasets_per_chunk):
            final_prompt = self.prompt_builder.build_question_generation_prompt(project_id, chunk, taxonomy_project, taxonomy_name)
            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different questions from this text chunk. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    item["result_type"] = "questions"
                    results.append(item)
            else:
                result["result_type"] = "questions"
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def generate_answers(self, project_id: int, chunk: str,
                         taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                         datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates multiple answers for a single chunk.
        """
        results = []
        for i in range(datasets_per_chunk):
            final_prompt = self.prompt_builder.build_answer_generation_prompt(project_id, chunk, taxonomy_project, taxonomy_name)
            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different answers from this text chunk. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    item["result_type"] = "answers"
                    results.append(item)
            else:
                result["result_type"] = "answers"
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def generate_summaries(self, project_id: int, chunk: str,
                           taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                           datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generates multiple summaries for a single chunk.
        """
        results = []
        for i in range(datasets_per_chunk):
            final_prompt = self.prompt_builder.build_summarization_prompt(project_id, chunk, taxonomy_project, taxonomy_name)
            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different summaries from this text chunk. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    item["result_type"] = "summary"
                    results.append(item)
            else:
                result["result_type"] = "summary"
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]
