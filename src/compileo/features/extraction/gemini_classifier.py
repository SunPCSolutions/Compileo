"""
This module provides a function to classify a text chunk based on a hierarchical taxonomy
using the Google Gemini API.
"""

import json
from google import genai
from google.genai import types
from typing import Dict, List, Any, Optional
from .context_models import DocumentContext, ChunkContext, HierarchicalCategory
from ...core.logging import get_logger

logger = get_logger(__name__)

def _classify_with_categories(
    text: str,
    categories: List[str],
    api_key: str,
    document_context: Optional[DocumentContext] = None,
    chunk_context: Optional[ChunkContext] = None,
    taxonomy: Optional[HierarchicalCategory] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    client = genai.Client(api_key=api_key)

    # Use provided model or default
    gemini_model = model if model is not None else "gemini-2.5-flash"

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
            context_info += f"\nPrevious Content: {' '.join(chunk_context.previous_chunks[-1:])}"
        if chunk_context.next_chunks:
            context_info += f"\nNext Content: {' '.join(chunk_context.next_chunks[:1])}"

    prompt = f"""
    Generate a hierarchical JSON classification for the following text based on the provided categories.
    Your response must be a single, valid JSON object and nothing else.
    {context_info}

    Text:
    {text}

    Categories:
    {categories}

    JSON:
    """

    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        response_text = response.text

        final_json = json.loads(response_text)

        if isinstance(final_json, dict):
            return final_json
        else:
            return {}

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error processing Gemini response: {e}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred with Gemini: {e}")
        return {}

def _generate_categories_with_gemini(text: str, num_categories: int, api_key: str, model: Optional[str] = None) -> List[str]:
    """
    Generates a list of high-level categories from a text chunk using the Gemini API.
    """
    client = genai.Client(api_key=api_key)

    # Use provided model or default
    gemini_model = model if model is not None else "gemini-2.5-flash"

    prompt = f"""
    Analyze the following text and generate {num_categories} high-level categories.
    Your response must be a single, valid JSON list of strings and nothing else.

    Text:
    {text}

    Categories:
    """

    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            },
        )
        response_text = response.text
        
        categories = json.loads(response_text)
        
        if isinstance(categories, list):
            return categories
        elif isinstance(categories, dict):
            # Look for a key that contains a list of categories
            for key, value in categories.items():
                if isinstance(value, list):
                    return value
        return []

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error processing Gemini response for category generation: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred with Gemini: {e}")
        return []
def classify_chunk(
    chunk_text: str,
    high_level_categories: Optional[List[str]],
    num_categories: int,
    api_key: Optional[str],
    document_context: Optional[DocumentContext] = None,
    chunk_context: Optional[ChunkContext] = None,
    taxonomy: Optional[HierarchicalCategory] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a hierarchical classification for a text chunk using the Gemini API with context awareness.

    Args:
        chunk_text (str): The text chunk to classify.
        high_level_categories (Optional[List[str]]): A list of high-level categories to guide the model.
        num_categories (int): The number of categories to generate.
        api_key (Optional[str]): The Google API key for Gemini.
        document_context (Optional[DocumentContext]): Document-level metadata and context.
        chunk_context (Optional[ChunkContext]): Context from adjacent chunks.
        taxonomy (Optional[HierarchicalCategory]): Hierarchical category taxonomy for classification.
        model (Optional[str]): The Gemini model to use. If None, uses default "gemini-2.5-flash".

    Returns:
        Dict[str, Any]: A dictionary representing the generated hierarchical classification.
    """
    if not api_key:
        raise ValueError("API key for Gemini must be provided.")

    # Use provided model or default
    gemini_model = model if model is not None else "gemini-2.5-flash"

    output = {
        "custom": {},
        "aigen": {}
    }

    if high_level_categories:
        output["custom"] = _classify_with_categories(
            chunk_text, high_level_categories, api_key, document_context, chunk_context, taxonomy, gemini_model
        )

    if num_categories > 0:
        aigen_categories = _generate_categories_with_gemini(chunk_text, num_categories, api_key, gemini_model)
        if aigen_categories:
            output["aigen"] = _classify_with_categories(
                chunk_text, aigen_categories, api_key, document_context, chunk_context, taxonomy, gemini_model
            )

    return output