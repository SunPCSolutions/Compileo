"""Constants and templates for dataset generation.

This module contains shared constants, JSON schemas, and prompt templates
used across the dataset generation module to eliminate code duplication.
"""

import json
from typing import Dict, Any

# =============================================================================
# Common Constraints
# =============================================================================

CONTENT_CONSTRAINTS = """CRITICAL CONSTRAINTS:
- ONLY use information and concepts that are explicitly present in the provided text chunk above.
- DO NOT draw from external knowledge, general knowledge, or information not contained in the text.
- DO NOT generate synthetic or hypothetical content.
- If the text does not contain sufficient information for a response, do not generate one.
- Base all content strictly on the information provided in the text chunk.
- For the reasoning field: provide a DETAILED explanation of the causal/logical reasoning behind the answer/output based on the text. DO NOT just cite "the text states...". Instead, elucidate the actual underlying mechanism, process, or logic described.
- If the text does not contain sufficient information to deduce detailed reasoning, leave the reasoning field empty (null or empty string). DO NOT hallucinate a reason."""

# =============================================================================
# JSON Schemas for LLM Responses
# =============================================================================

QA_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The generated question"},
        "answer": {"type": "string", "description": "The answer to the question"},
        "reasoning": {"type": ["string", "null"], "description": "Detailed explanation of the causal or logical reasoning behind the answer. Do not just cite the text. Leave empty/null if the text lacks sufficient information to explain the underlying logic."},
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Confidence score between 0.0 and 1.0"},
        "reasoning_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Array of specific logical steps leading to the answer"
        }
    },
    "required": ["question", "answer", "reasoning", "confidence_score", "reasoning_steps"],
    "additionalProperties": False
}

INSTRUCTION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string", "description": "A clear, actionable instruction"},
        "input": {"type": "string", "description": "Input context (can be empty)"},
        "output": {"type": "string", "description": "A comprehensive response to the instruction"},
        "reasoning": {"type": ["string", "null"], "description": "Detailed explanation of the causal or logical reasoning behind the output. Do not just cite the text. Leave empty/null if the text lacks sufficient information to explain the underlying logic."},
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Array of specific logical steps leading to the output"
        }
    },
    "required": ["instruction", "input", "output", "reasoning", "confidence_score", "reasoning_steps"],
    "additionalProperties": False
}

QUESTION_GENERATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of questions based on the content"
        },
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["questions", "confidence_score"],
    "additionalProperties": False
}

ANSWER_GENERATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of answers based on the content"
        },
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["answers", "confidence_score"],
    "additionalProperties": False
}

SUMMARIZATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Concise summary of the text"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key points from the text"
        },
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["summary", "key_points", "confidence_score"],
    "additionalProperties": False
}

# =============================================================================
# Array Schemas for AI-Determined Entry Count
# =============================================================================

QA_RESPONSE_ARRAY_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The generated question"},
            "answer": {"type": "string", "description": "The answer to the question"},
            "reasoning": {"type": ["string", "null"], "description": "Detailed explanation of the causal or logical reasoning behind the answer. Do not just cite the text. Leave empty/null if the text lacks sufficient information to explain the underlying logic."},
            "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of specific logical steps leading to the answer"
            }
        },
        "required": ["question", "answer", "reasoning", "confidence_score", "reasoning_steps"]
    },
    "description": "Array of question-answer pairs. Generate as many as appropriate for the content."
}

INSTRUCTION_RESPONSE_ARRAY_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "instruction": {"type": "string", "description": "A clear, actionable instruction"},
            "input": {"type": "string", "description": "Input context (can be empty)"},
            "output": {"type": "string", "description": "A comprehensive response to the instruction"},
            "reasoning": {"type": ["string", "null"], "description": "Detailed explanation of the causal or logical reasoning behind the output. Do not just cite the text. Leave empty/null if the text lacks sufficient information to explain the underlying logic."},
            "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of specific logical steps leading to the output"
            }
        },
        "required": ["instruction", "input", "output", "reasoning", "confidence_score", "reasoning_steps"]
    },
    "description": "Array of instruction-response pairs. Generate as many as appropriate for the content."
}

# =============================================================================
# Prompt Templates
# =============================================================================

def build_json_instructions(schema: Dict[str, Any], constraints: str = CONTENT_CONSTRAINTS,
                           multiple_items: int = 0, ai_decides: bool = False) -> str:
    """Build standardized JSON instructions for LLM prompts.
    
    Args:
        schema: The JSON schema to include in instructions
        constraints: Content constraints to include
        multiple_items: If > 0, add instruction to generate multiple items
        ai_decides: If True, let AI determine the number of entries
        
    Returns:
        Formatted instruction string
    """
    instructions = f"""
{constraints}

IMPORTANT: Provide your response as a valid machine-readable structure with exactly this schema:
{json.dumps(schema, indent=2)}

Return ONLY the raw object or array, no additional text."""
    
    if ai_decides:
        item_type = "question-answer pairs" if "question" in str(schema).lower() else "items"
        instructions += f"""\n\nGenerate as many {item_type} as appropriate based on the richness and diversity of the content.
Consider:
- The amount of distinct information in the text
- Different concepts, topics, or entities present
- Varying complexity levels suitable for different audiences
- Return 1-5 entries for sparse content, 5-10 for rich content, more for very comprehensive content.
Return them as a machine-readable array of objects."""
    elif multiple_items > 1:
        item_type = "question-answer pairs" if "question" in str(schema).lower() else "items"
        instructions += f"\n\nGenerate {multiple_items} different {item_type} based on the content provided above. Return them as a machine-readable array of objects."
    
    return instructions


def build_qa_prompt_template(chunk_text: str, base_prompt: str, datasets_per_chunk: int = 1) -> str:
    """Build a standardized QA generation prompt.
    
    Args:
        chunk_text: The text content to generate from
        base_prompt: The base prompt template
        datasets_per_chunk: Number of QA pairs to generate (0 = AI decides)
        
    Returns:
        Complete prompt string
    """
    if "{chunk}" in base_prompt:
        final_prompt = base_prompt.replace("{chunk}", chunk_text)
    else:
        final_prompt = f"Content: {chunk_text}\n\n{base_prompt}"
    
    # Add reasoning instructions for QA
    if "question" in final_prompt.lower() or "answer" in final_prompt.lower():
        if datasets_per_chunk == 0:
            final_prompt += build_json_instructions(QA_RESPONSE_ARRAY_SCHEMA, ai_decides=True)
        elif datasets_per_chunk == 1:
            final_prompt += build_json_instructions(QA_RESPONSE_SCHEMA, multiple_items=1)
        else:
            final_prompt += build_json_instructions(QA_RESPONSE_ARRAY_SCHEMA, multiple_items=datasets_per_chunk)
    
    return final_prompt


# =============================================================================
# Default Values
# =============================================================================

DEFAULT_DATASET_PARAMS = {
    "purpose": "general understanding",
    "audience": "general audience",
    "extraction_rules": "default",
    "dataset_format": "jsonl",
    "question_style": "factual",
    "answer_style": "comprehensive",
    "negativity_ratio": 0.1,
    "data_augmentation": "none",
    "custom_audience": "",
    "custom_purpose": "",
    "complexity_level": "intermediate",
    "domain": "general",
    "datasets_per_chunk": 3
}

# =============================================================================
# Generation Modes
# =============================================================================

GENERATION_MODES = {
    "question_and_answer": "qa",
    "question": "question",
    "answer": "answer",
    "summarization": "summarization",
    "instruction_following": "instruction_following",
    "custom": "custom"
}

# Strategy mapping for dispatcher
STRATEGY_METHOD_MAP = {
    "question": "generate_questions",
    "answer": "generate_answers",
    "summarization": "generate_summary",
    "instruction_following": "generate_instruction_following",
    "custom": "generate_with_custom_prompt"
}
