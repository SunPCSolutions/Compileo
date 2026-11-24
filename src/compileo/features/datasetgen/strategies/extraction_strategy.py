from typing import List, Dict, Any, Optional, Union
import json
from src.compileo.core.logging import get_logger
from .base_strategy import BaseGenerationStrategy
from ..constants import (
    QA_RESPONSE_SCHEMA, INSTRUCTION_RESPONSE_SCHEMA,
    build_json_instructions, CONTENT_CONSTRAINTS
)

logger = get_logger(__name__)


class ExtractionGenerationStrategy(BaseGenerationStrategy):
    """Strategy for generating datasets from extracted data with owner isolation."""
    
    def __init__(self, prompt_builder, llm_interaction, instruction_strategy):
        super().__init__(prompt_builder, llm_interaction)
        self.instruction_strategy = instruction_strategy

    def aggregate_entities_by_category(self, extracted_data: List[Dict[str, Any]],
                                     category_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Aggregate entities and texts by category across all extracted items."""
        category_map: Dict[str, Dict[str, List]] = {}
        
        # Log first item structure for debugging
        if extracted_data:
            first_item = extracted_data[0]
            metadata = first_item.get('metadata', {})
            logger.info(f"First extraction item keys: {list(first_item.keys())}")
            logger.info(f"First item metadata keys: {list(metadata.keys())}")
            if 'extracted_entities' in metadata:
                logger.info(f"Extracted entities keys: {list(metadata['extracted_entities'].keys())}")

        for item in extracted_data:
            metadata = item.get('metadata', {})
            
            if 'extracted_entities' not in metadata:
                logger.warning(f"Item missing 'extracted_entities' in metadata. Available keys: {list(metadata.keys())}")
                continue
                
            for category_id, category_data in metadata['extracted_entities'].items():
                category_name = category_data.get('category_name', category_id)
                category_id_str = str(category_id)
                
                # Filter by selected categories if provided
                if category_names:
                    if category_name not in category_names and category_id_str not in category_names:
                        continue

                texts = category_data.get('texts', [])
                entities = category_data.get('entities', [])

                if category_name not in category_map:
                    category_map[category_name] = {'texts': [], 'entities': []}
                
                if texts:
                    category_map[category_name]['texts'].extend(texts)
                if entities:
                    category_map[category_name]['entities'].extend(entities)

        # Create synthetic items for each category
        items_to_process = []
        for cat_name, data in category_map.items():
            if not data['texts'] and not data['entities']:
                continue
                
            items_to_process.append({
                'chunk_id': f"category_{cat_name}",
                'categories': [cat_name],
                'confidence': 1.0,
                'metadata': {
                    'extraction_type': 'aggregated',
                    'extracted_entities': {
                        cat_name: {
                            'category_name': cat_name,
                            'texts': data['texts'],
                            'entities': data['entities']
                        }
                    }
                },
                'entities': data['entities']
            })
            
        return items_to_process

    def generate_content_from_extracted_data(self, project_id: str, owner_id: str,
                                           prompt_name: str, extracted_item: Dict[str, Any],
                                           datasets_per_chunk: int = 3,
                                           custom_prompt: Optional[str] = None,
                                           generation_mode: str = "question and answer",
                                           category_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate content from extracted data (single item/category)."""
        results = []
        metadata = extracted_item.get('metadata', {})
        extraction_type = metadata.get('extraction_type', 'unknown')

        if 'extracted_entities' not in metadata:
            return results

        category_context = f"Focus on these categories: {', '.join(category_names)}. " if category_names else ""

        for category_id, category_data in metadata['extracted_entities'].items():
            category_name = category_data.get('category_name', category_id)
            
            if category_names and category_name not in category_names:
                continue
                
            texts = category_data.get('texts', [])
            entities = category_data.get('entities', [])

            if not texts and not entities:
                continue

            combined_text = "\n\n".join(texts)
            category_specific_context = f"Category: {category_name}. {category_context}"

            # For AI-determined mode (datasets_per_chunk == 0), make a single call
            if datasets_per_chunk == 0:
                if combined_text.strip():
                    final_prompt = self._build_prompt_from_text_content(
                        combined_text, generation_mode, custom_prompt, prompt_name,
                        project_id, owner_id, datasets_per_chunk, category_specific_context
                    )
                else:
                    final_prompt = self._build_prompt_from_entities(
                        entities, generation_mode, custom_prompt, prompt_name,
                        project_id, owner_id, datasets_per_chunk, category_specific_context
                    )

                result = self.llm_interaction.generate(final_prompt)
                # max_items=0 means no limit (AI-determined)
                processed = self._process_result(
                    result, "extract_adaptive", 0,
                    categories=[category_name],
                    confidence=extracted_item.get('confidence', 0.0),
                    extraction_type=extraction_type,
                    generation_mode=generation_mode
                )
                results.extend(processed)
                logger.info(f"AI-determined mode generated {len(processed)} entries from category {category_name}")
            else:
                # Fixed count mode: loop for specified number of items
                for i in range(datasets_per_chunk):
                    if combined_text.strip():
                        final_prompt = self._build_prompt_from_text_content(
                            combined_text, generation_mode, custom_prompt, prompt_name,
                            project_id, owner_id, datasets_per_chunk, category_specific_context
                        )
                    else:
                        final_prompt = self._build_prompt_from_entities(
                            entities, generation_mode, custom_prompt, prompt_name,
                            project_id, owner_id, datasets_per_chunk, category_specific_context
                        )

                    result = self.llm_interaction.generate(final_prompt)
                    processed = self._process_result(
                        result, "extract_adaptive", datasets_per_chunk,
                        categories=[category_name],
                        confidence=extracted_item.get('confidence', 0.0),
                        extraction_type=extraction_type,
                        generation_mode=generation_mode
                    )
                    results.extend(processed)

                    if len(results) >= datasets_per_chunk:
                        break
        
        # Only apply limit for fixed-count mode
        return results if datasets_per_chunk == 0 else results[:datasets_per_chunk]

    def _build_prompt_from_text_content(self, text_content: str, generation_mode: str,
                                       custom_prompt: Optional[str], prompt_name: str,
                                       project_id: str, owner_id: str, 
                                       datasets_per_chunk: int, 
                                       category_context: str = "") -> str:
        """Build prompt from extracted text content."""
        
        # Build base prompt based on generation mode
        if custom_prompt and custom_prompt.strip():
            if "{chunk}" in custom_prompt:
                base_prompt = custom_prompt.replace("{chunk}", text_content)
            else:
                base_prompt = f"Content: {text_content}\n\n" + custom_prompt
            base_prompt += "\n\nBase the generated content strictly on the provided text above."
        else:
            base_prompt = self._get_mode_specific_prompt(
                generation_mode, project_id, owner_id, text_content, prompt_name, datasets_per_chunk
            )

        # Add category context if provided
        if category_context:
            base_prompt = f"{category_context}\n\n{base_prompt}"

        # Add JSON instructions based on generation mode
        base_prompt += self._get_json_instructions(generation_mode, datasets_per_chunk)
        
        return base_prompt

    def _build_prompt_from_entities(self, entities: List[str], generation_mode: str,
                                   custom_prompt: Optional[str], prompt_name: str,
                                   project_id: str, owner_id: str,
                                   datasets_per_chunk: int,
                                   category_context: str = "") -> str:
        """Build prompt from extracted entities."""
        entity_text = f"Focus on these concepts: {', '.join(entities)}" if entities else "various concepts"

        if custom_prompt and custom_prompt.strip():
            if "{chunk}" in custom_prompt:
                final_prompt = custom_prompt.replace("{chunk}", entity_text)
            else:
                final_prompt = f"Content: {entity_text}\n\n" + custom_prompt
        else:
            final_prompt = self.prompt_builder.build_prompt(
                project_id=project_id,
                owner_id=owner_id,
                prompt_name=prompt_name,
                chunk_text=entity_text,
                taxonomy_project=None,
                taxonomy_name=None,
                datasets_per_chunk=datasets_per_chunk
            )
            final_prompt += f"\n\nGenerate educational content about these concepts: {entity_text}."
            if datasets_per_chunk > 1:
                final_prompt += f"\n\nGenerate {datasets_per_chunk} different question-answer pairs about these concepts."

        if category_context:
            final_prompt = f"{category_context}\n\n{final_prompt}"

        if datasets_per_chunk == 0:
            from ..constants import QA_RESPONSE_ARRAY_SCHEMA
            final_prompt += build_json_instructions(QA_RESPONSE_ARRAY_SCHEMA, ai_decides=True)
        else:
            final_prompt += build_json_instructions(QA_RESPONSE_SCHEMA, multiple_items=datasets_per_chunk)
        
        return final_prompt

    def _get_mode_specific_prompt(self, generation_mode: str, project_id: str, owner_id: str,
                                 text_content: str, prompt_name: str,
                                 datasets_per_chunk: int) -> str:
        """Get the appropriate prompt builder method based on generation mode."""
        if generation_mode == "question":
            return self.prompt_builder.build_question_generation_prompt(
                project_id, owner_id, text_content, None, None
            )
        elif generation_mode == "answer":
            return self.prompt_builder.build_answer_generation_prompt(
                project_id, owner_id, text_content, None, None
            )
        elif generation_mode == "summarization":
            return self.prompt_builder.build_summarization_prompt(
                project_id, owner_id, text_content, None, None
            )
        elif generation_mode == "instruction_following":
            return self.instruction_strategy.build_instruction_following_prompt(
                text_content, "explain", None, None
            )
        else:  # default or "question_and_answer"
            return self.prompt_builder.build_prompt(
                project_id=project_id,
                owner_id=owner_id,
                prompt_name=prompt_name,
                chunk_text=text_content,
                taxonomy_project=None,
                taxonomy_name=None,
                datasets_per_chunk=datasets_per_chunk
            )

    def _get_json_instructions(self, generation_mode: str, datasets_per_chunk: int) -> str:
        """Get JSON instructions based on generation mode."""
        ai_decides = (datasets_per_chunk == 0)
        
        if generation_mode == "instruction_following":
            if ai_decides:
                from ..constants import INSTRUCTION_RESPONSE_ARRAY_SCHEMA
                return build_json_instructions(INSTRUCTION_RESPONSE_ARRAY_SCHEMA, ai_decides=True)
            elif datasets_per_chunk == 1:
                return build_json_instructions(INSTRUCTION_RESPONSE_SCHEMA, multiple_items=1)
            else:
                from ..constants import INSTRUCTION_RESPONSE_ARRAY_SCHEMA
                return build_json_instructions(INSTRUCTION_RESPONSE_ARRAY_SCHEMA, multiple_items=datasets_per_chunk)
        elif generation_mode in ["question", "answer", "summarization", "question and answer", "qa"]:
            # These modes already have specialized instructions in their templates
            return ""
        else:
            if ai_decides:
                from ..constants import QA_RESPONSE_ARRAY_SCHEMA
                return build_json_instructions(QA_RESPONSE_ARRAY_SCHEMA, ai_decides=True)
            elif datasets_per_chunk == 1:
                return build_json_instructions(QA_RESPONSE_SCHEMA, multiple_items=1)
            else:
                from ..constants import QA_RESPONSE_ARRAY_SCHEMA
                return build_json_instructions(QA_RESPONSE_ARRAY_SCHEMA, multiple_items=datasets_per_chunk)
