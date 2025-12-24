"""
This module provides a function to classify a text chunk based on a hierarchical taxonomy
using the OpenAI API (ChatGPT).
"""

import json
from typing import Dict, List, Any, Optional
from .context_models import DocumentContext, ChunkContext, HierarchicalCategory
from ...core.settings import backend_settings
from ...core.logging import get_logger

logger = get_logger(__name__)

def _classify_with_categories(
    text: str,
    categories: List[str],
    api_key: str,
    model: Optional[str] = None,
    document_context: Optional[DocumentContext] = None,
    chunk_context: Optional[ChunkContext] = None,
    taxonomy: Optional[HierarchicalCategory] = None
) -> Dict[str, Any]:
    """
    Classifies text using provided categories via OpenAI API with context awareness.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Error: 'openai' package not installed.")
        return {}

    client = OpenAI(api_key=api_key)

    # Use provided model or get from settings (default to gpt-4o)
    openai_model = model if model is not None else backend_settings.get_classification_openai_model()

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

    # Convert categories list to string representation for the prompt
    categories_str = json.dumps(categories)

    system_prompt = f"""You are an expert classifier specializing in extracting and organizing information from texts. Your role is to analyze documents and narratives to identify and extract relevant details for specified categories, ensuring accuracy and relevance.{context_info}

Extract information from the provided text for each of the following categories:
{categories_str}

Guidelines:
1. Read the text thoroughly and understand the context.
2. Consider the document context and adjacent content to better understand the full picture.
3. For each provided category, scan the text for any mentions, descriptions, or implications related to that category.
4. Extract specific details, measurements, observations, or key facts that directly pertain to each category.
5. If a category has no relevant information in the text, use an empty string or null value for that key.
6. Summarize or list the extracted information concisely.
7. Ensure the extracted information is directly supported by the text.

You must return a valid JSON object where keys are the category names and values are the extracted information."""

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to analyze:\n{text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        response_content = response.choices[0].message.content
        if not response_content:
            return {}

        return json.loads(response_content)

    except Exception as e:
        logger.error(f"Error interacting with OpenAI API: {e}")
        return {}


def generate_categories(
    chunk_text: str, 
    num_categories: int, 
    api_key: str, 
    model: Optional[str] = None, 
    exclude_categories: Optional[List[str]] = None
) -> List[str]:
    """
    Generates a list of high-level categories from a text chunk using OpenAI API.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Error: 'openai' package not installed.")
        return []

    client = OpenAI(api_key=api_key)
    
    # Use provided model or get from settings
    openai_model = model if model is not None else backend_settings.get_classification_openai_model()

    exclude_text = ""
    if exclude_categories:
        exclude_text = f" Exclude these categories: {', '.join(exclude_categories)}."

    system_prompt = f"""You are an expert text analyst specializing in content classification. Your task is to analyze text and generate high-level categories that capture the core themes and topics relevant to the content.

Analyze the user provided text and generate exactly {num_categories} high-level categories that are relevant to the content but distinct from any excluded ones.{exclude_text}

Return only a JSON object with a single key "categories" containing a list of strings, like this:
{{
  "categories": ["category1", "category2", "category3"]
}}"""

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        response_content = response.choices[0].message.content
        if not response_content:
            return []

        data = json.loads(response_content)
        return data.get("categories", [])

    except Exception as e:
        logger.error(f"Error generating categories with OpenAI API: {e}")
        return []


def classify_chunk(
    chunk_text: str,
    api_key: str,
    model: Optional[str] = None,
    high_level_categories: Optional[List[str]] = None,
    num_categories: int = 0,
    document_context: Optional[DocumentContext] = None,
    chunk_context: Optional[ChunkContext] = None,
    taxonomy: Optional[HierarchicalCategory] = None
) -> Dict[str, Any]:
    """
    Generates a hierarchical classification for a text chunk using OpenAI API with context awareness.

    Args:
        chunk_text (str): The text chunk to classify.
        api_key (str): The OpenAI API key.
        model (Optional[str]): The OpenAI model to use.
        high_level_categories (Optional[List[str]]): A list of high-level categories to guide the model.
        num_categories (int): The number of categories to generate for aigen.
        document_context (Optional[DocumentContext]): Document-level metadata and context.
        chunk_context (Optional[ChunkContext]): Context from adjacent chunks.
        taxonomy (Optional[HierarchicalCategory]): Hierarchical category taxonomy for classification.

    Returns:
        Dict[str, Any]: A dictionary with 'custom' and 'aigen' classifications.
    """
    if not api_key:
        logger.error("Error: OpenAI API key not provided.")
        return {"custom": {}, "aigen": {}}

    output = {
        "custom": {},
        "aigen": {}
    }

    if high_level_categories:
        output["custom"] = _classify_with_categories(
            chunk_text, high_level_categories, api_key, model, document_context, chunk_context, taxonomy
        )

    if num_categories > 0:
        aigen_categories = generate_categories(chunk_text, num_categories, api_key, model, exclude_categories=high_level_categories)
        if aigen_categories:
            output["aigen"] = _classify_with_categories(
                chunk_text, aigen_categories, api_key, model, document_context, chunk_context, taxonomy
            )

    return output