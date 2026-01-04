from typing import List, Dict, Any, Optional, Union
import json
from src.compileo.core.logging import get_logger
from .base_strategy import BaseGenerationStrategy
from .instruction_strategy import InstructionGenerationStrategy
from ..prompt_builder import PromptBuilder
from ..llm_interaction import LLMInteraction

logger = get_logger(__name__)

class ExtractionGenerationStrategy(BaseGenerationStrategy):
    """
    Strategy for generating datasets from extracted data.
    """
    def __init__(self, prompt_builder: PromptBuilder, llm_interaction: LLMInteraction,
                 instruction_strategy: InstructionGenerationStrategy):
        super().__init__(prompt_builder, llm_interaction)
        self.instruction_strategy = instruction_strategy

    def generate_qa_for_extracted_item(self, project_id: int, prompt_name: str, extracted_item: Dict[str, Any],
                                       taxonomy_project: Optional[str] = None, taxonomy_name: Optional[str] = None,
                                       datasets_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Generate Q&A pairs for a single extracted item using its categorized content.
        """
        results = []
        chunk_text = extracted_item.get('chunk_text', '')
        categories = extracted_item.get('categories', [])

        # Create enhanced prompt using categories for better context
        category_context = f"Content categories: {', '.join(categories)}" if categories else ""

        for i in range(datasets_per_chunk):
            # Build prompt with category context
            final_prompt = self.prompt_builder.build_prompt(
                project_id=project_id,
                prompt_name=prompt_name,
                chunk_text=chunk_text,
                taxonomy_project=taxonomy_project,
                taxonomy_name=taxonomy_name,
                datasets_per_chunk=datasets_per_chunk
            )

            # Add category context to the prompt
            if category_context:
                final_prompt = f"{category_context}\n\n{final_prompt}"

            # Modify prompt to request multiple items if needed
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs from this categorized content. Return them as a JSON array."

            result = self.llm_interaction.generate(final_prompt)

            # Handle multiple results
            if isinstance(result, list):
                for item in result[:datasets_per_chunk]:
                    item["categories"] = categories
                    item["confidence"] = extracted_item.get('confidence', 0.0)
                    item["source_type"] = "extraction"
                    results.append(item)
            else:
                result["categories"] = categories
                result["confidence"] = extracted_item.get('confidence', 0.0)
                result["source_type"] = "extraction"
                results.append(result)

            if len(results) >= datasets_per_chunk:
                break

        return results[:datasets_per_chunk]

    def aggregate_entities_by_category(self, extracted_data: List[Dict[str, Any]], 
                                     category_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Aggregate entities and texts by category across all extracted items.
        Returns a list of synthetic items representing aggregated categories.
        """
        category_map = {}  # category_name -> {'texts': [], 'entities': []}

        for item in extracted_data:
            metadata = item.get('metadata', {})

            if 'extracted_entities' in metadata:
                for category_id, category_data in metadata['extracted_entities'].items():
                    category_name = category_data.get('category_name', category_id)

                    # Filter by selected categories if provided
                    if category_names and category_name not in category_names:
                        continue

                    texts = category_data.get('texts', [])
                    entities = category_data.get('entities', [])

                    # Initialize category in map if not present
                    if category_name not in category_map:
                        category_map[category_name] = {'texts': [], 'entities': []}
                    
                    # Aggregate texts if present
                    if texts:
                        category_map[category_name]['texts'].extend(texts)
                    
                    # Aggregate entities if present
                    if entities:
                        category_map[category_name]['entities'].extend(entities)

        # Create synthetic items for each category to process
        items_to_process = []
        for cat_name, data in category_map.items():
            texts = data['texts']
            entities = data['entities']
            
            # Skip if no content found for this category
            if not texts and not entities:
                continue
                
            # Create a synthetic item that focuses on this single category
            synthetic_item = {
                'chunk_id': f"category_{cat_name}",
                'categories': [cat_name],
                'confidence': 1.0,
                'metadata': {
                    'extraction_type': 'aggregated',
                    'extracted_entities': {
                        cat_name: {
                            'category_name': cat_name,
                            'texts': texts,
                            'entities': entities
                        }
                    }
                },
                'entities': entities  # Also set at top level for compatibility
            }
            items_to_process.append(synthetic_item)
            
        return items_to_process

    def generate_content_from_extracted_data(self, project_id: int, prompt_name: str, extracted_item: Dict[str, Any],
                                             datasets_per_chunk: int = 3, custom_prompt: Optional[str] = None,
                                             generation_mode: str = "question and answer", category_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate content from extracted data (single item/category), handling both NER and full text extraction types.
        """
        results = []
        categories = extracted_item.get('categories', [])
        metadata = extracted_item.get('metadata', {})

        # Determine extraction type and available content
        extraction_type = metadata.get('extraction_type', 'unknown')

        # Build category context from provided category_names if available
        category_context = ""
        if category_names:
            category_context = f"Focus on these categories: {', '.join(category_names)}. "

        if 'extracted_entities' in metadata:
            for category_id, category_data in metadata['extracted_entities'].items():
                category_name = category_data.get('category_name', category_id)
                if category_names and category_name not in category_names:
                    continue
                texts = category_data.get('texts', [])
                entities = category_data.get('entities', [])

                if not texts and not entities:
                    continue

                # Combine texts for this category
                combined_text = "\n\n".join(texts)

                # Generate for this category's content
                for i in range(datasets_per_chunk):
                    # Build prompt with category-specific context
                    category_specific_context = f"Category: {category_name}. {category_context}"

                    if combined_text.strip():
                        final_prompt = self._build_prompt_from_text_content(
                            combined_text, generation_mode, custom_prompt, prompt_name, project_id, datasets_per_chunk, category_specific_context
                        )
                    else:
                        # Fallback to entity-based if no text
                        final_prompt = self._build_prompt_from_entities(
                            extracted_item, generation_mode, custom_prompt, prompt_name, project_id, datasets_per_chunk, category_specific_context
                        )

                    result = self.llm_interaction.generate(final_prompt)

                    # Handle extract mode: if result is dict with answer as JSON string, expand it
                    expanded_items = []
                    if isinstance(result, dict) and isinstance(result.get('answer'), str):
                        try:
                            parsed = json.loads(result['answer'])
                            if isinstance(parsed, list):
                                for item in parsed[:datasets_per_chunk]:
                                    if item and isinstance(item, dict):
                                        item_copy = item.copy()
                                        item_copy["categories"] = [category_name]
                                        item_copy["confidence"] = extracted_item.get('confidence', 0.0)
                                        item_copy["extraction_type"] = extraction_type
                                        item_copy["source_type"] = "extract_adaptive"
                                        expanded_items.append(item_copy)
                        except (json.JSONDecodeError, KeyError):
                            pass

                    if expanded_items:
                        results.extend(expanded_items)
                    else:
                        # Normal handling
                        if isinstance(result, list):
                            for item in result[:datasets_per_chunk]:
                                if item and isinstance(item, dict):
                                    item_copy = item.copy()
                                    item_copy["categories"] = [category_name]
                                    item_copy["confidence"] = extracted_item.get('confidence', 0.0)
                                    item_copy["extraction_type"] = extraction_type
                                    item_copy["source_type"] = "extract_adaptive"
                                    
                                    # Map instruction following fields for fine-tuning appropriateness
                                    if generation_mode == "instruction following":
                                        item_copy.setdefault("instruction", item_copy.get("instruction", ""))
                                        item_copy.setdefault("input", item_copy.get("input", ""))
                                        item_copy.setdefault("output", item_copy.get("output", ""))
                                        
                                    results.append(item_copy)
                        elif isinstance(result, dict) and result:
                            item_copy = result.copy()
                            item_copy["categories"] = [category_name]
                            item_copy["confidence"] = extracted_item.get('confidence', 0.0)
                            item_copy["extraction_type"] = extraction_type
                            item_copy["source_type"] = "extract_adaptive"
                            
                            # Map specialized fields based on mode and clean up unrelated fields
                            final_item = {
                                "categories": [category_name],
                                "confidence": extracted_item.get('confidence', 0.0),
                                "extraction_type": extraction_type,
                                "source_type": "extract_adaptive",
                                "reasoning": item_copy.get("reasoning", ""),
                                "confidence_score": item_copy.get("confidence_score", 0.0),
                                "reasoning_steps": item_copy.get("reasoning_steps", []),
                                "generation_mode": generation_mode
                            }

                            if generation_mode == "instruction following":
                                final_item.update({
                                    "instruction": item_copy.get("instruction", ""),
                                    "input": item_copy.get("input", ""),
                                    "output": item_copy.get("output", "")
                                })
                            elif generation_mode == "question":
                                # Support both singular and plural from LLM
                                val = item_copy.get("question") or item_copy.get("questions")
                                if isinstance(val, list):
                                    final_item["questions"] = val
                                elif val:
                                    final_item["question"] = val
                                else:
                                    final_item["question"] = ""
                            elif generation_mode == "answer":
                                # Support both singular and plural from LLM
                                val = item_copy.get("answer") or item_copy.get("answers")
                                if isinstance(val, list):
                                    final_item["answers"] = val
                                elif val:
                                    final_item["answer"] = val
                                else:
                                    final_item["answer"] = ""
                            elif generation_mode == "summarization":
                                final_item.update({
                                    "summary": item_copy.get("summary", ""),
                                    "key_points": item_copy.get("key_points", [])
                                })
                            else:
                                # Default Q&A mode
                                final_item.update({
                                    "question": item_copy.get("question", ""),
                                    "answer": item_copy.get("answer", "")
                                })
                                
                            results.append(final_item)

                        if len(results) >= datasets_per_chunk:
                            break
        
        return results[:datasets_per_chunk]

    def _build_prompt_from_text_content(self, text_content: str, generation_mode: str,
                                       custom_prompt: Optional[str], prompt_name: str,
                                       project_id: Union[int, str], datasets_per_chunk: int, category_context: str = "") -> str:
        """Build prompt from extracted text content."""
        base_prompt = ""
        if custom_prompt and custom_prompt.strip():
            if "{chunk}" in custom_prompt:
                base_prompt = custom_prompt.replace("{chunk}", text_content)
            else:
                base_prompt = f"Content: {text_content}\n\n" + custom_prompt
            # Add instruction to base on content
            base_prompt += "\n\nBase the generated content strictly on the provided text above. Use any examples only as format guidelines. Do not copy the examples; create new content derived from the provided text."
        else:
            # Convert project_id to int for prompt_builder if it's a numeric string
            proj_id_int = int(project_id) if isinstance(project_id, str) and project_id.isdigit() else project_id
            
            if generation_mode == "question":
                base_prompt = self.prompt_builder.build_question_generation_prompt(
                    proj_id_int, text_content, None, None, datasets_per_chunk # type: ignore
                )
            elif generation_mode == "answer":
                base_prompt = self.prompt_builder.build_answer_generation_prompt(
                    proj_id_int, text_content, None, None, datasets_per_chunk # type: ignore
                )
            elif generation_mode == "summarization":
                base_prompt = self.prompt_builder.build_summarization_prompt(
                    proj_id_int, text_content, None, None, datasets_per_chunk # type: ignore
                )
            elif generation_mode == "instruction following":
                base_prompt = self.instruction_strategy.build_instruction_following_prompt(
                    text_content, "explain", None, None
                )
            else:  # default or "question and answer"
                base_prompt = self.prompt_builder.build_prompt(
                    project_id=proj_id_int, # type: ignore
                    prompt_name=prompt_name,
                    chunk_text=text_content,
                    taxonomy_project=None,
                    taxonomy_name=None,
                    datasets_per_chunk=datasets_per_chunk
                )

        # Add category context if provided
        if category_context:
            base_prompt = f"{category_context}\n\n{base_prompt}"

        # Add JSON instructions
        if generation_mode == "instruction following":
            json_instructions = """\n\n
        IMPORTANT: You must respond with a valid machine-readable structure with exactly this schema:
        {
          "instruction": "A clear, actionable instruction related to the content",
          "input": "",
          "output": "A comprehensive, helpful response to the instruction",
          "reasoning": "Brief explanation of your reasoning",
          "confidence_score": 0.8,
          "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
        }

        CRITICAL CONSTRAINTS:
        - ONLY use information and concepts that are explicitly present in the provided content.
        - DO NOT draw from external knowledge, general knowledge, or information not contained in the content.
        - DO NOT generate synthetic or hypothetical content.
        - If the content does not contain sufficient information for an instruction-response pair, return an empty array [].
        - Base all instructions and responses strictly on the content provided.

        Return ONLY the raw object or array, no additional text or explanation.
        """
        elif generation_mode in ["question", "answer", "summarization"]:
            # These modes already have specialized instructions in their templates
            json_instructions = ""
        else:
            json_instructions = """\n\n
        IMPORTANT: You must respond with a valid machine-readable structure with exactly this schema:
        {
          "question": "The question you generated based on the content",
          "answer": "The answer to the question",
          "reasoning": "Brief explanation of your reasoning",
          "confidence_score": 0.8,
          "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
        }

        CRITICAL CONSTRAINTS:
        - ONLY use information explicitly present in the provided content.
        - Generate questions that are directly answerable from the content provided.
        - Base answers strictly on the content provided.
        - If the content doesn't support generating meaningful questions and answers, return an empty array [].

        Return ONLY the raw object or array, no additional text or explanation.
        """
        
        final_prompt = base_prompt + json_instructions
        
        if datasets_per_chunk > 1 and generation_mode not in ["question", "answer", "summarization"]:
            if generation_mode == "instruction following":
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different instruction-response pairs based on the content provided above. Return them as a machine-readable array of objects."
            else:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs based on the content provided above. Return them as a machine-readable array of objects."
        
        return final_prompt

    def _build_prompt_from_entities(self, extracted_item: Dict[str, Any], generation_mode: str,
                                   custom_prompt: Optional[str], prompt_name: str,
                                   project_id: int, datasets_per_chunk: int, category_context: str = "") -> str:
        """Build prompt from extracted entities (for NER extractions)."""
        entities = extracted_item.get('entities', [])
        entity_text = ", ".join(entities) if entities else "various concepts"

        if generation_mode == "question":
            return self.prompt_builder.build_question_generation_prompt(
                project_id, f"Focus on these concepts: {entity_text}", None, None, datasets_per_chunk
            )
        elif generation_mode == "answer":
            return self.prompt_builder.build_answer_generation_prompt(
                project_id, f"Focus on these concepts: {entity_text}", None, None, datasets_per_chunk
            )
        elif generation_mode == "summarization":
            return self.prompt_builder.build_summarization_prompt(
                project_id, f"Focus on these concepts: {entity_text}", None, None, datasets_per_chunk
            )
        elif generation_mode == "instruction following":
            return self.instruction_strategy.build_instruction_following_prompt(
                f"Focus on these concepts: {entity_text}", "explain", None, None
            )
        else:  # default or "question and answer"
            if custom_prompt and custom_prompt.strip():
                if "{chunk}" in custom_prompt:
                    final_prompt = custom_prompt.replace("{chunk}", f"Focus on these concepts: {entity_text}")
                else:
                    final_prompt = f"Content: Focus on these concepts: {entity_text}\n\n" + custom_prompt
                # Add category context if provided
                if category_context:
                    final_prompt = f"{category_context}\n\n{final_prompt}"
                if generation_mode == "instruction following":
                    json_instructions = """

IMPORTANT: You must respond with a valid JSON object (or array of objects) with exactly this structure:
{
  "instruction": "A clear, actionable instruction related to the content",
  "input": "",
  "output": "A comprehensive, helpful response to the instruction",
  "reasoning": "Brief explanation of your reasoning",
  "confidence_score": 0.8,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}

CRITICAL CONSTRAINTS:
- ONLY use information and concepts that are explicitly present in the provided content.
- DO NOT draw from external knowledge, general knowledge, or information not contained in the content.
- DO NOT generate synthetic or hypothetical content.
- If the content does not contain sufficient information for an instruction-response pair, return an empty array [].
- Base all instructions and responses strictly on the content provided.

Return ONLY the JSON object or array, no additional text or explanation."""
                else:
                    json_instructions = """

IMPORTANT: You must respond with a valid machine-readable structure with exactly this schema:
{
  "question": "The question you generated based on the content",
  "answer": "The answer to the question",
  "reasoning": "Brief explanation of your reasoning",
  "confidence_score": 0.8,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"]
}

CRITICAL CONSTRAINTS:
- ONLY use information explicitly present in the provided content.
- Generate questions that are directly answerable from the content provided.
- Base answers strictly on the content provided.
- If the content doesn't support generating meaningful questions and answers, return an empty array [].

Return ONLY the raw object or array, no additional text or explanation."""

                final_prompt += json_instructions

                if datasets_per_chunk > 1 and generation_mode not in ["question", "answer", "summarization"]:
                    if generation_mode == "instruction following":
                        final_prompt += f"\n\nGenerate {datasets_per_chunk} different instruction-response pairs based on the content provided above. Return them as a machine-readable array of objects."
                    else:
                        final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs based on the content provided above. Return them as a machine-readable array of objects."
                return final_prompt
            else:
                final_prompt = self.prompt_builder.build_prompt(
                    project_id=project_id,
                    prompt_name=prompt_name,
                    chunk_text=f"Focus on these concepts: {entity_text}",
                    taxonomy_project=None,
                    taxonomy_name=None
                )
                # Add category context if provided
                if category_context:
                    final_prompt = f"{category_context}\n\n{final_prompt}"
                final_prompt += f"\n\nGenerate educational content about these concepts: {entity_text}. Create questions and answers that help users understand these concepts."
                if datasets_per_chunk > 1:
                    final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs about these concepts. Return them as a JSON array."
                return final_prompt
