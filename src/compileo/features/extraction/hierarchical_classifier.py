"""
This module provides functions to classify and validate a text chunk based on a hierarchical taxonomy
using Large Language Models.
"""

import json
from google import genai
from typing import Dict, List, Any, Optional
from .ollama_classifier import classify_chunk as classify_with_ollama
from ...core.settings import backend_settings
from ...core.logging import get_logger

logger = get_logger(__name__)

def validate_classification(chunk_text: str, classification: Dict[str, Any], high_level_categories: List[str], api_key: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Validates and refines a hierarchical classification using the Gemini LLM.

    Args:
        chunk_text (str): The text chunk to classify.
        classification (Dict[str, Any]): The initial classification from the Ollama model.
        high_level_categories (List[str]): A list of high-level categories to guide the LLM.
        api_key (str, optional): The Google API key. Defaults to None.
        model (Optional[str]): The Gemini model to use. If None, uses default from settings.

    Returns:
        Dict[str, Any]: A dictionary representing the validated and refined hierarchical classification.
    """
    # Create client with API key if provided
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    # Use provided model or get from settings
    gemini_model_name = model if model is not None else backend_settings.get_classification_gemini_model()

    prompt = f"""
    Review the following JSON classification based on the text chunk and the provided high-level categories.
    Ensure the output strictly adheres to the specified categories.
    Return ONLY the refined JSON object.

    Text Chunk:
    ---
    {chunk_text}
    ---

    High-Level Categories:
    ---
    {high_level_categories}
    ---

    Initial Classification:
    ---
    {json.dumps(classification, indent=2)}
    ---
    """

    try:
        response = client.models.generate_content(
            model=gemini_model_name,
            contents=prompt
        )
        
        # Clean the response to extract only the JSON part.
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        
        parsed_json = json.loads(cleaned_response_text)
        
        if isinstance(parsed_json, dict):
            return parsed_json

        return {}

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing Gemini response: {e}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred with Gemini: {e}")
        return {}

def classify_and_validate(chunk_text: str, high_level_categories: List[str], api_key: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrates the two-step classification and validation process.

    Args:
        chunk_text (str): The text chunk to classify.
        high_level_categories (List[str]): A list of high-level categories to guide the LLM.
        api_key (str, optional): The Google API key. Defaults to None.
        model (Optional[str]): The Gemini model to use. If None, uses default from settings.

    Returns:
        Dict[str, Any]: A dictionary representing the validated hierarchical classification.
    """
    # Step 1: Generate initial classification with Ollama
    logger.debug("--- Generating initial classification with Ollama ---")
    initial_classification = classify_with_ollama(chunk_text, model, high_level_categories)
    logger.debug("--- Ollama classification ---")
    logger.debug(json.dumps(initial_classification, indent=2))

    if not initial_classification:
        return {}

    # Step 2: Validate and refine with Gemini
    logger.debug("--- Validating and refining classification with Gemini ---")
    validated_classification = validate_classification(chunk_text, initial_classification, high_level_categories, api_key, model)
    logger.debug("--- Gemini classification ---")
    logger.debug(json.dumps(validated_classification, indent=2))


    return validated_classification