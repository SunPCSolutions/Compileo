"""
This module provides a function to classify a text chunk based on a hierarchical taxonomy
using an Ollama model.
"""

import json
import re
import requests
from typing import Dict, List, Any, Optional
from .aigen_classifier import generate_categories
from .context_models import DocumentContext, ChunkContext, HierarchicalCategory, ClassificationResult
from ...core.logging import get_logger

logger = get_logger(__name__)

def _sanitize_json(text: str) -> str:
    """
    Extracts and sanitizes a JSON string from a text response.
    """
    # Find the start and end of the JSON object
    start_index = text.find('{')
    end_index = text.rfind('}')
    if start_index == -1 or end_index == -1:
        raise ValueError("No JSON object found in the response")
    
    json_text = text[start_index:end_index+1]
    
    # Remove trailing commas
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
    
    return json_text

def _classify_with_categories(
    text: str,
    categories: List[str],
    model: Optional[str] = None,
    document_context: Optional[DocumentContext] = None,
    chunk_context: Optional[ChunkContext] = None,
    taxonomy: Optional[HierarchicalCategory] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    from ...core.settings import backend_settings
    api_url = f"{backend_settings.get_ollama_base_url()}/api/generate"

    # Build context information
    context_info = ""
    if document_context:
        context_parts = []
        if document_context.title:
            context_parts.append(f"Document Title: {document_context.title}")
        if document_context.summary:
            context_parts.append(f"Document Summary: {document_context.summary}")
        if document_context.author:
            context_parts.append(f"Author: {document_context.author}")
        if document_context.keywords:
            context_parts.append(f"Keywords: {', '.join(document_context.keywords)}")
        if context_parts:
            context_info += "\n\nDocument Context:\n" + "\n".join(context_parts)

    if chunk_context and (chunk_context.previous_chunks or chunk_context.next_chunks):
        context_info += "\n\nAdjacent Content for Context:"
        if chunk_context.previous_chunks:
            context_info += f"\nPrevious Content: {' '.join(chunk_context.previous_chunks[-1:])}"  # Last previous chunk
        if chunk_context.next_chunks:
            context_info += f"\nNext Content: {' '.join(chunk_context.next_chunks[:1])}"  # First next chunk

    # Convert categories list to string representation
    categories_str = json.dumps(categories)

    prompt_template = f"""[INST] You are an expert medical classifier specializing in extracting and organizing clinical information from medical texts. Your role is to analyze patient records, diagnostic reports, and medical narratives to identify and extract relevant details for specified categories, ensuring accuracy and clinical relevance.{context_info}

Extract information from the following text for each of the provided categories.

Think step by step:
1. Read the text thoroughly and understand the clinical context, including patient symptoms, diagnoses, treatments, and outcomes.
2. Consider the document context and adjacent content to better understand the full clinical picture.
3. For each provided category, scan the text for any mentions, descriptions, or implications related to that category.
4. Extract specific details, measurements, observations, or key facts that directly pertain to each category.
5. If a category has no relevant information in the text, use an empty string or null value for that key.
6. Summarize or list the extracted information concisely, maintaining medical terminology where appropriate.
7. Ensure the extracted information is directly supported by the text and clinically meaningful.
8. Organize the results into a clean JSON object with category names as keys and extracted information as values.

Few-shot examples:
Example 1:
Text: "Patient admitted with pneumonia, fever 101.5°F, cough productive of green sputum. Chest X-ray shows right lower lobe infiltrate. Started on ceftriaxone and azithromycin."
Categories: ["Symptoms", "Diagnosis", "Treatment"]
Output: {{"Symptoms": "Fever 101.5°F, productive cough with green sputum", "Diagnosis": "Pneumonia, right lower lobe infiltrate on chest X-ray", "Treatment": "Ceftriaxone and azithromycin"}}

Example 2:
Text: "45-year-old male with hypertension, BP 160/95 mmHg. Prescribed lisinopril 10mg daily. Follow-up in 2 weeks."
Categories: ["Demographics", "Vital Signs", "Medications"]
Output: {{"Demographics": "45-year-old male", "Vital Signs": "BP 160/95 mmHg", "Medications": "Lisinopril 10mg daily"}}

Text: {text}

Categories: {categories_str}

Return only a valid JSON object, nothing else. [/INST]"""
    prompt = prompt_template

    # Use provided model or get from settings
    ollama_model = model if model is not None else backend_settings.get_classification_ollama_model()

    # Default options from settings
    default_options = {
        "num_ctx": backend_settings.get_classification_ollama_num_ctx(),
        "temperature": backend_settings.get_classification_ollama_temperature(),
        "repeat_penalty": backend_settings.get_classification_ollama_repeat_penalty(),
        "top_p": backend_settings.get_classification_ollama_top_p(),
        "top_k": backend_settings.get_classification_ollama_top_k(),
        "num_predict": backend_settings.get_classification_ollama_num_predict()
    }

    seed = backend_settings.get_classification_ollama_seed()
    if seed is not None:
        default_options["seed"] = seed

    # Merge provided options with defaults
    merged_options = {**default_options, **(options or {})}

    request_data = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": merged_options
    }
    try:
        response = requests.post(api_url, json=request_data)
        response.raise_for_status()

        response_json = response.json()
        response_content = response_json.get("response", "{}")

        final_json = json.loads(response_content)

        if isinstance(final_json, dict):
            return final_json
        else:
            return {}

    except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error interacting with Ollama: {e}")
        return {}

def classify_chunk(
    chunk_text: str,
    model: Optional[str] = None,
    high_level_categories: Optional[List[str]] = None,
    num_categories: int = 0,
    document_context: Optional[DocumentContext] = None,
    chunk_context: Optional[ChunkContext] = None,
    taxonomy: Optional[HierarchicalCategory] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates a hierarchical classification for a text chunk using an Ollama model with context awareness.

    Args:
        chunk_text (str): The text chunk to classify.
        model (Optional[str]): The Ollama model to use. If None, uses default from settings.
        high_level_categories (Optional[List[str]]): A list of high-level categories to guide the model.
        num_categories (int): The number of categories to generate.
        document_context (Optional[DocumentContext]): Document-level metadata and context.
        chunk_context (Optional[ChunkContext]): Context from adjacent chunks.
        taxonomy (Optional[HierarchicalCategory]): Hierarchical category taxonomy for classification.
        options: Optional Ollama API options to override defaults

    Returns:
        Dict[str, Any]: A dictionary representing the generated hierarchical classification.
    """
    output = {
        "custom": {},
        "aigen": {}
    }

    # Add context usage flags
    if document_context is not None or chunk_context is not None or taxonomy is not None:
        output["metadata"] = {
            "document_context_used": document_context is not None,
            "chunk_context_used": chunk_context is not None,
            "taxonomy_used": taxonomy is not None
        }

    # Use provided options or construct from settings
    if options is None:
        from ...core.settings import backend_settings
        options = {
            "num_ctx": backend_settings.get_classification_ollama_num_ctx(),
            "temperature": backend_settings.get_classification_ollama_temperature(),
            "repeat_penalty": backend_settings.get_classification_ollama_repeat_penalty(),
            "top_p": backend_settings.get_classification_ollama_top_p(),
            "top_k": backend_settings.get_classification_ollama_top_k(),
            "num_predict": backend_settings.get_classification_ollama_num_predict()
        }
        seed = backend_settings.get_classification_ollama_seed()
        if seed is not None:
            options["seed"] = seed

    if high_level_categories:
        output["custom"] = _classify_with_categories(
            chunk_text, high_level_categories, model, document_context, chunk_context, taxonomy, options
        )

    if num_categories > 0:
        aigen_categories = generate_categories(chunk_text, num_categories, exclude_categories=high_level_categories)
        if aigen_categories:
            output["aigen"] = _classify_with_categories(
                chunk_text, aigen_categories, model, document_context, chunk_context, taxonomy, options
            )

    return output